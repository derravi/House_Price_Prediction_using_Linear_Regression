from pydantic import BaseModel,Field
from typing import Annotated

#Make The Pydantic Model For User Input
class UserInput(BaseModel):
    Area_sqft:Annotated[float,Field(...,gt=0,description="Enter the Area of the House (in Square foot).",examples=[123.25])]
    Bedrooms:Annotated[int,Field(...,gt=0,description="Enter the Total number of Bedrooms.",examples=[4])]
    Bathrooms:Annotated[float,Field(...,gt=0,description="Enter the Total Number of Bathrooms.",examples=[1.0])]
    Floors:Annotated[float,Field(...,gt=0,description="Enter the Number of Floors.",examples=[1])]
    Year_Built:Annotated[int,Field(...,gt=0,description="Enter the Birth Year of the House.",examples=[2019])]
