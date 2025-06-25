import pickle
import os
import sys
import numpy as np
import SimpleITK as sitk


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter. 1 m = 1000 mm
    """

    def __init__(self, data):
        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = (
            data["DSD"] / 1000
        )  # Distance Source to Detector      (m) x射线发射源到x射线接收器之间的距离
        self.DSO = (
            data["DSO"] / 1000
        )  # Distance Source Origin        (m) 发射源到起点之间的距离

        # Detector parameters
        self.nDetector = np.array(
            data["nDetector"]
        )  # number of pixels              (px)
        self.dDetector = (
            np.array(data["dDetector"]) / 1000
        )  # size of each pixel            (m)
        self.sDetector = (
            self.nDetector * self.dDetector
        )  # total size of the detector    (m)

        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = (
            np.array(data["dVoxel"]) / 1000
        )  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = (
            np.array(data["offOrigin"]) / 1000
        )  # Offset of image from origin   (m)
        self.offDetector = (
            np.array(data["offDetector"]) / 1000
        )  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data[
            "accuracy"
        ]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


path = "./data/chest_50.pickle"
with open(path, "rb") as handle:
    data = pickle.load(handle)
    # stx()

geo = ConeGeometry(data)  # 把数据处理成ConeGeometry
print(
    f"DSO= {geo.DSO}, DSD={geo.DSD}, nVoxel={geo.nVoxel}, dVoxel={geo.dVoxel}, sVoxel={geo.sVoxel}"
)  # noqa: E501
print(
    f"nDetector={geo.nDetector}, dDetector={geo.dDetector}, sDetector={geo.sDetector}"
)  # noqa: E501
image = data["image"]
save_nifti(image, "./chest_50.nii.gz")
