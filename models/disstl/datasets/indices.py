"""
Modified from https://github.com/microsoft/torchgeo/blob/main/torchgeo/transforms/indices.py
"""
import torch
import torch.nn as nn

_EPSILON = 1e-10


def ndvi(red: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Different Vegetation Index (NDVI).
    Args:
        red: tensor containing red band
        nir: tensor containing nir band
    Returns:
        tensor containing computed NDVI values
    """
    return (nir - red) / ((nir + red) + _EPSILON)


def savi(red: torch.Tensor, nir: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    """Compute Soil Adjusted Vegetation Factor (SAVI).
    Args:
        red: tensor containing red band
        nir: tensor containing nir band
        factor: soil brightness correction factor
    Returns:
        tensor containing computed SAVI values
    """
    return ((nir - red) * (1 + factor)) / (nir + red + factor + _EPSILON)


def ndwi(green: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Different Water Index (NDWI).
    Args:
        green: tensor containing green band
        nir: tensor containing nir band
    Returns:
        tensor containing computed NDWI values
    """
    return (green - nir) / ((green + nir) + _EPSILON)


def mndwi(green: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute Modified Normalized Different Water Index (MNDWI).
    Args:
        green: tensor containing green band
        swir: tensor containing swir band
    Returns:
        tensor containing computed MNDWI values
    """
    return (green - swir) / ((green + swir) + _EPSILON)


def ndbi(nir: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Different Built-up Index (NDBI).
    Args:
        nir: tensor containing nir band
        swir: tensor containing swir band
    Returns:
        tensor containing computed NDBI values
    """
    return (swir - nir) / ((swir + nir) + _EPSILON)


def baei(red: torch.Tensor, green: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute Built-up Area Extraction Index (BAEI).
    Args:
        red: tensor containing red band
        green: tensor containing green band
        swir: tensor containing swir band
    Returns:
        tensor containing computed BAEI values
    """
    return (red + 0.3) / ((green + swir) + _EPSILON)


def bui(red: torch.Tensor, nir: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute Built-up Index (BUI).
    Args:
        red: tensor containing red band
        nir: tensor containing nir band
        swir: tensor containing swir band
    Returns:
        tensor containing computed BUI values
    """
    return ((swir - nir) / ((swir + nir) + _EPSILON)) / ((nir - red) / ((nir + red) + _EPSILON))


def nbi(red: torch.Tensor, nir: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute New Built-up Index (NBI).
    Args:
        red: tensor containing red band
        nir: tensor containing nir band
        swir: tensor containing swir band
    Returns:
        tensor containing computed NBI values
    """
    return (red - swir) / (nir + _EPSILON)


def nbai(green: torch.Tensor, swir1: torch.Tensor, swir2: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Built-up Area Index (NBAI).
    Args:
        green: tensor containing green band
        swir1: tensor containing swir1 band
        swir2: tensor containing swir2 band
    Returns:
        tensor containing computed NBAI values
    """
    return ((swir2 - swir1) / (green + _EPSILON)) / ((swir2 + swir1) / (green + _EPSILON))


def brba(red: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute Band ration for Built-up Area (BRBA).
    Args:
        red: tensor containing red band
        swir: tensor containing swir band
    Returns:
        tensor containing computed BRBA values
    """
    return red / (swir + _EPSILON)


def ibi(red: torch.Tensor, green: torch.Tensor, nir: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute Index based Built-up Index (IBI).
    Args:
        red: tensor containing red band
        green: tensor containing green band
        nir: tensor containing nir band
        swir: tensor containing swir band
    Returns:
        tensor containing computed IBI values
    """
    x = ndbi(swir, nir)
    y = (savi(red, nir) + mndwi(green, swir)) / 2
    return (x - y) / (x + y + _EPSILON)


def mbai(green: torch.Tensor, nir: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Compute Modified Built-up Area Index (MBAI).
    Args:
        green: tensor containing green band
        nir: tensor containing nir band
        swir: tensor containing swir band
    Returns:
        tensor containing computed MBAI values
    """
    return (nir + (1.57 * green) + (2.4 * swir)) / (1 + nir + _EPSILON)


def ndcci(green: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Difference Concrete Condition Index (NDCCI).
    Args:
        green: tensor containing green band
        nir: tensor containing nir band
    Returns:
        tensor containing computed NDCCI values
    """
    return (nir - green) / (nir + green + _EPSILON)


class NDVI(nn.Module):
    """Normalized Difference Vegetation Index (NDVI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/0034-4257(79)90013-0
    """

    ordered_bands = ["red", "nir"]

    def __init__(self, index_red: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_nir = index_nir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for NDVI.

        Args:
            x: a single data sample

        Returns:
            an image representing NDVI
        """
        return ndvi(red=x[:, self.index_red], nir=x[:, self.index_nir])


class SAVI(nn.Module):
    """Soil Adjusted Vegetation Index (SAVI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/0034-4257(88)90106-X
    """

    ordered_bands = ["red", "nir"]

    def __init__(self, index_red: int, index_nir: int, factor: float = 0.5) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_nir = index_nir
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for SAVI.

        Args:
            x: a single data sample

        Returns:
            an image representing SAVI
        """
        return savi(red=x[:, self.index_red], nir=x[:, self.index_nir], factor=self.factor)


class NDWI(nn.Module):
    """Normalized Difference Water Index (NDWI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431169608948714
    """

    ordered_bands = ["green", "nir"]

    def __init__(self, index_green: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.index_green = index_green
        self.index_nir = index_nir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for NDWI.

        Args:
            x: a single data sample

        Returns:
            an image representing NDWI
        """
        return ndwi(green=x[:, self.index_green], nir=x[:, self.index_nir])


class MNDWI(nn.Module):
    """Modified Normalized Difference Water Index (MNDWI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431160600589179
    """

    ordered_bands = ["green", "swir"]

    def __init__(self, index_green: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_swir: index of the Short Wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_green = index_green
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for MNDWI.

        Args:
            x: a single data sample

        Returns:
            an image representing NDWI
        """
        return mndwi(green=x[:, self.index_green], swir=x[:, self.index_swir])


class NDBI(nn.Module):
    """Normalized Difference Built-up Index (NDBI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431160304987
    """

    ordered_bands = ["nir", "swir"]

    def __init__(self, index_nir: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_nir = index_nir
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for NDBI.

        Args:
            x: a single data sample

        Returns:
            an image representing NDBI
        """
        return ndbi(nir=x[:, self.index_nir], swir=x[:, self.index_swir])


class BAEI(nn.Module):
    """Built-up Area Extraction Index (BAEI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1007/s12524-015-0460-6
    """

    ordered_bands = ["red", "green", "swir"]

    def __init__(self, index_red: int, index_green: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_green: index of the Green band in the image
            index_swir: index of the Short Wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_green = index_green
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for BAEI.

        Args:
            x: a single data sample

        Returns:
            an image representing BAEI
        """
        return baei(red=x[:, self.index_red], green=x[:, self.index_green], swir=x[:, self.index_swir])


class BUI(nn.Module):
    """Built-up Index (BUI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431161.2010.481681
    """

    ordered_bands = ["red", "nir", "swir"]

    def __init__(self, index_red: int, index_nir: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_nir = index_nir
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for BUI.

        Args:
            x: a single data sample

        Returns:
            an image representing BUI
        """
        return bui(red=x[:, self.index_red], nir=x[:, self.index_nir], swir=x[:, self.index_swir])


class NBI(nn.Module):
    """New Built-up Index (NBI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1109/GEOINFORMATICS.2010.5567823
    """

    ordered_bands = ["red", "nir", "swir"]

    def __init__(self, index_red: int, index_nir: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_nir = index_nir
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for NBI.

        Args:
            x: a single data sample

        Returns:
            an image representing NBI
        """
        return nbi(red=x[:, self.index_red], nir=x[:, self.index_nir], swir=x[:, self.index_swir])


class NBAI(nn.Module):
    """Normalized Built-up Area Index (NBAI).

    If you use this index in your research, please cite the following paper:

    * https://www.omicsonline.org/scientific-reports/JGRS-SR136.pdf
    """

    ordered_bands = ["green", "swir1", "swir2"]

    def __init__(self, index_green: int, index_swir1: int, index_swir2: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Red band in the image
            index_swir1: index of the Short-wave Infrared (SWIR1) band in the image
            index_swir2: index of the Short-wave Infrared (SWIR2) band in the image
        """
        super().__init__()
        self.index_green = index_green
        self.index_swir1 = index_swir1
        self.index_swir2 = index_swir2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for NBAI.

        Args:
            x: a single data sample

        Returns:
            an image representing NBAI
        """
        return nbai(green=x[:, self.index_green], swir1=x[:, self.index_swir1], swir2=x[:, self.index_swir2])


class BRBA(nn.Module):
    """Band ration for Built-up Area (BRBA).

    If you use this index in your research, please cite the following paper:

    * https://www.omicsonline.org/scientific-reports/JGRS-SR136.pdf
    """

    ordered_bands = ["red", "swir"]

    def __init__(self, index_red: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for BRBA.

        Args:
            x: a single data sample

        Returns:
            an image representing BRBA
        """
        return brba(red=x[:, self.index_red], swir=x[:, self.index_swir])


class IBI(nn.Module):
    """Index based Built-up Index (IBI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431160802039957
    """

    ordered_bands = ["red", "green", "nir", "swir"]

    def __init__(self, index_red: int, index_green: int, index_nir: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: index of the Red band in the image
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_red = index_red
        self.index_green = index_green
        self.index_nir = index_nir
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for IBI.

        Args:
            x: a single data sample

        Returns:
            an image representing IBI
        """
        return ibi(
            red=x[:, self.index_red], green=x[:, self.index_green], nir=x[:, self.index_nir], swir=x[:, self.index_swir]
        )


class MBAI(nn.Module):
    """Modified Built-up Area Index (MBAI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1007/s12524-018-0895-7
    """

    ordered_bands = ["green", "nir", "swir"]

    def __init__(self, index_green: int, index_nir: int, index_swir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
            index_swir: index of the Short-wave Infrared (SWIR) band in the image
        """
        super().__init__()
        self.index_green = index_green
        self.index_nir = index_nir
        self.index_swir = index_swir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for MBAI.

        Args:
            x: a single data sample

        Returns:
            an image representing MBAI
        """
        return mbai(green=x[:, self.index_green], nir=x[:, self.index_nir], swir=x[:, self.index_swir])


class NDCCI(nn.Module):
    """Normalized Difference Concrete Condition Index (NDCCI).

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1117/1.JRS.10.025021
    """

    ordered_bands = ["green", "nir"]

    def __init__(self, index_green: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__()
        self.index_green = index_green
        self.index_nir = index_nir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a band for NDCCI.

        Args:
            x: a single data sample

        Returns:
            an image representing NDCCI
        """
        return ndcci(green=x[:, self.index_green], nir=x[:, self.index_nir])
