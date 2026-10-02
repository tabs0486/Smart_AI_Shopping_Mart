from pydantic import BaseModel, Field
from typing import List


class ShoppingRequest(BaseModel):
    category: str
    product: str
    maximum_budget: float
    condition: str
    delivery_city: str
    payment_method: str
    delivery_deadline_days: int
    ingredients: str = ""
    specifications: List[str] = Field(default_factory=list)
