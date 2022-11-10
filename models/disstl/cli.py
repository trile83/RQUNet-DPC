import torch  # noqa
import multiprocessing

import gc
import json
import logging
import os
from pathlib import Path
from shutil import copytree, move
from shapely.geometry import mapping
from typing import Dict, List, Optional

import click
import rasterio  # noqa F401

from disstl.annotations.region import Region
from disstl.cli_opts import AOI_TYPE, CRS_TYPE, JSON_DICT_TYPE, GlobalOpts, RESAMPLING_TYPE
from disstl.commons import configure_logging
from disstl.commons.aws.s3 import s3_get_file, s3_get_json, s3_put_dir, s3_put_file
from disstl.imagery.stac import stac_query_source, stac_s3_source
from disstl.utils.debug import timing

_logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], auto_envvar_prefix="DISSTL")
pass_global_opts = click.make_pass_decorator(GlobalOpts)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--work-path",
    type=click.Path(exists=False, file_okay=False, writable=True),
    help="Optional directory to use for local processing, defaults to a temporary dir",
    default=Path.home() / ".disstl",
)
@click.option("--keep-work/--no-keep-work", default=False)
@click.option("--debug/--no-debug", default=False)
@click.option(
    "--gc-threshold",
    help="GC generation threshold values (3 ints separated by commas)",
    default="350,5,5",
    show_default=True,
)
@click.pass_context
def main(ctx, work_path, keep_work, debug, gc_threshold):
    work_path = Path(work_path)
    work_path.mkdir(parents=True, exist_ok=True)
    configure_logging()

    # gc threshold defaults are (700, 10, 10) for gen0, 1, 2
    _logger.debug("setting gc threshold to %s", gc_threshold)
    gen0, gen1, gen2 = (int(g) for g in gc_threshold.split(","))
    gc.set_threshold(gen0, gen1, gen2)

    ctx.obj = ctx.with_resource(GlobalOpts(work_path, keep_work, debug))


@main.command(help="Run inferencing over an AOI")
@click.option("--aoi", default=None, type=AOI_TYPE)
@click.option("--time-period")
@click.option("--output-loc", required=True)
@click.option("--device", default="cpu")
@click.option("--chip-shape", type=click.IntRange(min=16), default=32, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--num-workers", type=click.IntRange(min=1))
@click.option("--num-prefetch", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--threshold-seg", type=click.FloatRange(min=0.0, max=1.0), default=0.9, show_default=True)
@click.option("--min-area-threshold", type=click.FloatRange(min=0.0), default=5000.0, show_default=True)
@click.option(
    "--stride",
    type=click.IntRange(min=-1),
    default=-1,
    show_default=True,
    help="Set to -1 for stride to be automatically calculated.",
)
@click.option("--temporal-repeats", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--input-cube-loc", default=None)
@click.option("--input-region-file-uri")
@click.option("--input-stac-list-uri")
@click.option("--model-loc", default="~/.disstl/models/latest", show_default=True)
@click.option("--seg-model-loc")
@click.option("--cb-model-loc")
@click.option("--manifest-loc")
@click.option(
    "--type",
    "--inference-type",
    "inference_type",
    required=True,
    type=click.Choice(["seg_nobas"], case_sensitive=False),
    default="seg_nobas",
    show_default=True,
)
@click.option(
    "--preview/--no-preview",
    default=False,
    show_default=True,
    help="Whether preview images and MP4s should be generated",
)
@click.option(
    "--include-parameters/--no-include-parameters",
    default=False,
    show_default=True,
    help="Whether or not to include the CLI parameters in the resulting site geojsons.",
)
@click.option(
    "--cube-crs",
    default=None,
    type=CRS_TYPE,
    help="Optional output CRS of cube as An EPSG (eg. 'epsg:4326'), PROJ json, or WKT string if cube is being built. "
    "Defaults to utm zone proj, of centroid",
)
@click.option(
    "--cube-resolution",
    default=None,
    type=float,
    help="Optional output resolution of the cube (in units of cube CRS (meters if cube-crs is not set).",
)
@click.option(
    "--resampling",
    default="bilinear",
    type=RESAMPLING_TYPE,
    help="Resampling to use when building data cube.",
)
@click.option(
    "--ckpt-type",
    default="last",
    required=True,
    type=click.Choice(["best", "last"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--run_id",
    help="A unique identifier for this inference run. In practice this is built by airflow, but any string is "
    "valid here",
)
@pass_global_opts
def run(
    global_opts,
    aoi,
    time_period,
    output_loc,
    device,
    chip_shape,
    batch_size,
    num_workers,
    num_prefetch,
    threshold_seg,
    min_area_threshold,
    stride,
    temporal_repeats,
    input_cube_loc,
    input_stac_list_uri,
    input_region_file_uri,
    model_loc,
    seg_model_loc,
    cb_model_loc,
    manifest_loc,
    inference_type,
    preview,
    include_parameters,
    cube_crs,
    cube_resolution,
    resampling,
    ckpt_type,
    run_id,
):
    if input_region_file_uri:
        # arguments that end in "json" may contain a trailing space when coming from airflow, hence .strip()
        region_json = s3_get_json(input_region_file_uri.strip())
        region = Region.from_geojson(region_json)
        time_period = time_period or region.period_str
        aoi = aoi or region.geometry
        region_id = region.region_id
    else:
        region = aoi
        region_id = "region"

    if input_cube_loc is None:
        assert time_period is not None or input_stac_list_uri is not None, (
            "Must define at least (--input-cube-loc)"
            " or (--input_stac_list_uri) or (--input_region_file_uri)"
            " or (--aoi and --time-period)"
        )
        if input_stac_list_uri is None:
            stac_items = stac_query_source(
                endpoint="https://earth-search.aws.element84.com/v0",
                bbox=list(aoi.bounds),
                period=time_period,
                collections=["sentinel-s2-l2a-cogs"],
            )
        else:
            stac_items = stac_s3_source(input_stac_list_uri)

        local_cube_path = _create_hdf5(
            global_opts,
            stac_items=stac_items,
            aoi=aoi,
            period=time_period,
            crs=cube_crs,
            resolution=cube_resolution,
            resampling=resampling,
        )
    elif input_cube_loc.startswith("s3://"):
        local_cube_path = global_opts.work_path / "cube" / input_cube_loc.split("/")[-1]
        local_cube_path.parent.mkdir(parents=True, exist_ok=True)
        s3_get_file(input_cube_loc, local_cube_path)
    else:
        local_cube_path = Path(input_cube_loc).expanduser()

    _logger.info(f"Performing inference on {local_cube_path}")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Local import to avoid slow click help generation
    from disstl import inference
    from disstl.inference.datasets import EmptyHDF5Error

    try:
        output_path, output_region = inference.run_inference(
            global_opts,
            device,
            local_cube_path,
            global_opts.work_path / region_id,
            inference_type,
            chip_shape=chip_shape,
            batch_size=batch_size,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            threshold_seg=threshold_seg,
            min_area_threshold=min_area_threshold,
            stride=stride,
            # temporal_repeats=temporal_repeats,
            include_parameters=include_parameters,
            disstl_parameters={
                "aoi": mapping(aoi) if aoi else None,
                "time_period": time_period,
                "output_loc": output_loc,
                "device": device,
                "chip_shape": chip_shape,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "num_prefetch": num_prefetch,
                "threshold_seg": threshold_seg,
                "min_area_threshold": min_area_threshold,
                "stride": stride,
                "input_cube_loc": input_cube_loc,
                "input_stac_list_uri": input_stac_list_uri,
                "input_region_file_uri": input_region_file_uri,
                "model_loc": model_loc,
                "seg_model_loc": seg_model_loc,
                "cb_model_loc": cb_model_loc,
                "manifest_loc": manifest_loc,
                "inference_type": inference_type,
                "preview": preview,
                "temporal_repeats": temporal_repeats,
                "ckpt_type": ckpt_type,
                "run_id": run_id,
            },
            ckpt_type=ckpt_type,
            model_loc=model_loc,
            seg_model_loc=seg_model_loc,
            cb_model_loc=cb_model_loc,
            region=region,
            run_id=run_id,
        )
    except EmptyHDF5Error:
        _logger.warning("Inference received an empty HDF5 as input. No inference has been performed.")
        return

    if preview:
        from disstl.imagery.hdf5 import make_previews

        make_previews(local_cube_path, output_path / "preview", output_region)

    _deliver_result(output_path, output_loc, manifest_loc=manifest_loc)
    _logger.info(f"Saved inference results to {output_loc}")


@main.command(help="Create cube data files for use by inference, training")
@click.option("--aoi", type=AOI_TYPE, help="An Area of Interest, (Geojson Geom, WKT, or json array bounds)")
@click.option("--time-period", help="A time period to use when querying (in STAC datetime format)")
@click.option(
    "--region-id",
    help="A region ID to use for AOI, and if --annotate is supplied, from which to build annotation masks",
)
@click.option("--output-loc", required=True)
@click.option(
    "--output-crs",
    default=None,
    type=CRS_TYPE,
    help="Optional output CRS of cube as An EPSG, PROJ json, or WKT string. If not provided, the UTM zone of AOI "
    "centroid will be used",
)
@click.option(
    "--output-resolution",
    default=None,
    type=float,
    help="Optional output resolution of the cube (in units of cube CRS (meters if output-crs is not set).",
)
@click.option(
    "--resampling",
    default="bilinear",
    type=RESAMPLING_TYPE,
    help="Resampling to use when building data cube.",
)
@click.option("--annotate/--no-annotate", default=False, help="Whether site annotation masks should be included")
@click.option("--pan-sharpen/--no-pan-sharpen", default=False, help="Whether to attempt pan sharpening if available")
@click.option(
    "--stac-endpoint",
    default="https://earth-search.aws.element84.com/v0",
    help="The base uri of the STAC catalog services to use when querying",
)
@click.option(
    "--stac-header",
    "stac_headers",
    multiple=True,
    help="a header to add to all STAC service interactions in the form KEY=VALUE (can be supplied multiple times)",
)
@click.option(
    "--stac-collection",
    "stac_collections",
    multiple=True,
    default=["sentinel-s2-l2a-cogs"],
    help="The STAC collection to search (can be supplied multiple times to search multiple collections)",
)
@click.option("--stac-query", type=JSON_DICT_TYPE, help="A json dict containing a custom STAC query")
@click.option("--region-file-loc", help="A S3 uri or file path to a specific region file")
@click.option("--input-stac-list-uri")
@click.option(
    "--max-cloud-cover",
    default=50,
    show_default=True,
    help="The maximum eo:cloud_cover to allow when processing STAC items",
)
@click.option("--preview-loc", help="An optional location where preview imagery and MP4s should be written")
@pass_global_opts
def etl(global_opts, **kwargs):
    if kwargs.get("region_file_loc"):
        # arguments that end in "json" may contain a trailing space when coming from airflow, hence .strip()
        region_json = s3_get_json(kwargs.get("region_file_loc").strip())
        region = Region.from_geojson(region_json)
    else:
        region_id = kwargs.get("region_id")
        region = Region.from_s3_annotations(region_id) if region_id else None

    annotate = region is not None and kwargs.get("annotate") is True

    aoi = kwargs.get("aoi") or region and region.geometry
    time_period = kwargs.get("time_period") or region and region.period_str

    assert aoi and time_period, (
        "Must define at least (--region-id) or (--region-file-loc) or " "(--aoi and --time-period)"
    )

    if kwargs.get("input_stac_list_uri"):
        stac_items = stac_s3_source(kwargs.get("input_stac_list_uri"))
    else:
        stac_items = stac_query_source(
            endpoint=kwargs["stac_endpoint"],
            headers=_stac_headers_to_dict(kwargs.get("stac_headers")),
            bbox=list(aoi.bounds),
            period=time_period,
            collections=kwargs["stac_collections"],
            sortby=[{"field": "properties.datetime", "direction": "asc"}],
        )
    hdf5_path = _create_hdf5(
        global_opts,
        stac_items=stac_items,
        aoi=aoi,
        crs=kwargs.get("output_crs"),
        time_period=time_period,
        region=region,
        annotate=annotate,
        pan_sharpen=kwargs.get("pan_sharpen"),
        max_cloud_cover=kwargs.get("max_cloud_cover"),
        resolution=kwargs.get("output_resolution"),
        resampling=kwargs.get("resampling"),
    )

    preview_loc = kwargs.get("preview_loc")
    if preview_loc:
        from disstl.imagery.hdf5 import make_previews

        local_preview_path = global_opts.work_path / "preview"
        make_previews(hdf5_path, local_preview_path, region)
        _deliver_result(local_preview_path, preview_loc)

    _deliver_result(hdf5_path, kwargs["output_loc"])


@main.command(help="Extract imagery and metadata from cube data files")
@click.option("-i", "--input-loc", required=True, help="Local path or S3 uri to input cube .hdf5 file")
@click.option("-o", "--output-loc", required=True, help="Local path or S3 uri prefix for extracted files")
@click.option("--preview/--no-preview", default=True, help="Whether preview images and MP4s should be generated")
@click.option("--tif/--no-tif", default=True, help="Whether full resolution TIF files should be generated")
@pass_global_opts
def extract(global_opts: GlobalOpts, input_loc, output_loc, preview, tif):
    if input_loc.startswith("s3://"):
        local_cube_path = global_opts.work_path / input_loc.split("/")[-1]
        local_cube_path.parent.mkdir(parents=True, exist_ok=True)
        s3_get_file(input_loc, local_cube_path)
    else:
        local_cube_path = Path(input_loc).expanduser()

    _logger.info(f"extracting cube {local_cube_path}")
    output_path = global_opts.work_path / "cube"

    # Local import to avoid slow click help generation
    from disstl.imagery.hdf5 import extract_hdf5

    extract_hdf5(local_cube_path, output_path, gen_preview=preview, extract_tifs=tif)
    _deliver_result(output_path, output_loc)


@main.command(help="List Cube metadata")
@click.option("-i", "--input-loc", required=True, help="Local path or S3 uri to input cube .hdf5 file")
@pass_global_opts
def info(global_opts: GlobalOpts, input_loc):
    if input_loc.startswith("s3://"):
        local_cube_path = global_opts.work_path / input_loc.split("/")[-1]
        local_cube_path.parent.mkdir(parents=True, exist_ok=True)
        s3_get_file(input_loc, local_cube_path)
    else:
        local_cube_path = Path(input_loc).expanduser()

    # Local import to avoid slow click help generation
    from disstl.imagery.hdf5 import get_metadata

    metadata = get_metadata(local_cube_path)
    click.echo(json.dumps(metadata, default=str, indent=2))


@timing(end_msg="delivery complete", logger=_logger)
def _deliver_result(src_path: Path, dst: str, manifest_loc=None):
    if not src_path.exists():
        return
    _logger.info("delivering final result to %s", dst)
    files_written = []
    if dst.startswith("s3://"):
        if src_path.is_file():
            s3_put_file(src_path, dst)
            files_written = [dst]
        else:
            files_written = s3_put_dir(src_path, dst)
    else:
        dst_path = Path(dst)
        if src_path != dst_path:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_file():
                move(src_path, dst_path)
                files_written = [dst_path]
            else:
                files_written = copytree(src_path, dst_path, dirs_exist_ok=True)
                for root, _, files in os.walk(dst_path):
                    files_written = [os.path.join(root, file) for file in files]
    if manifest_loc:
        local_manifest = os.path.join(src_path, "MANIFEST")
        with open(local_manifest, "w") as manifest:
            manifest.write("\n".join(files_written))
        _deliver_result(Path(local_manifest), manifest_loc)


def _stac_headers_to_dict(stac_headers: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not stac_headers:
        return None
    headers = {}
    for header in stac_headers:
        idx = header.index("=")
        if idx and len(header) > idx + 2:
            key = header[0:idx]
            value = header[idx + 1 :]
            headers[key] = value
    return headers if headers else None


def _create_hdf5(global_opts, **kwargs) -> Path:

    # Local imports to avoid slow click help generation
    from disstl.imagery.cube import Cube
    from disstl.imagery.hdf5 import export_hdf5

    cube = Cube.build(
        stac_items=kwargs["stac_items"],
        path=global_opts.work_path / "cube",
        aoi=kwargs.get("aoi"),
        crs=kwargs.get("crs"),
        region=kwargs.get("region"),
        annotate=kwargs.get("annotate"),
        pan_sharpen=kwargs.get("pan_sharpen"),
        max_cloud_cover=kwargs.get("max_cloud_cover"),
        resolution=kwargs.get("resolution"),
        resampling=kwargs.get("resampling"),
    )
    hdf5_path = global_opts.work_path / "region.hdf5"
    export_hdf5(cube, hdf5_path)
    return hdf5_path


if __name__ == "__main__":
    main()
