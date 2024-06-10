from langchain.tools import tool
from pydantic import Field, BaseModel
from PIL import Image
import cv2
import numpy as np
IMAGE_PATH = r"D:\fastaimodelmaker\Data\glioma_tumor\G_1.jpg"
class ImageInspect(BaseModel):
    file_path: str = Field(
        description="Must be a valid path to an image"
    )

@tool("inspect_image_properties", args_schema=ImageInspect)
def inspect_image_properties(file_path: str) -> str:
    """Tells you dimensions of image in this exact order: Height, Width, Channels"""
    if not file_path:
        return(
            "Please provide a valid file path"
        )
    img = cv2.imread(file_path)

    img_np = np.asarray(img)

    dimensions = f"Image Dimensions are: {img_np.shape}"


    return dimensions

if __name__ == "__main__":
    print(inspect_image_properties.run(r"D:\fastaimodelmaker\Data\glioma_tumor\G_18.jpg"))

