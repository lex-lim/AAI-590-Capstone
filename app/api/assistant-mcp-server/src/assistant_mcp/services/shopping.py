"""Shopping list service."""

from typing import Optional


class ShoppingListService:
    """Service for shopping list operations."""
    
    def __init__(self):
        # In-memory storage
        self.items = {}
    
    async def get_list(
        self, 
        category: Optional[str] = None,
        checked: Optional[bool] = None
    ) -> str:
        """Get shopping list items."""
        if not self.items:
            return "Shopping list is empty"
        
        filtered_items = [
            item for item in self.items.values()
            if (category is None or item.get("category") == category)
            and (checked is None or item.get("checked", False) == checked)
        ]
        
        if not filtered_items:
            filter_desc = []
            if category:
                filter_desc.append(f"category '{category}'")
            if checked is not None:
                filter_desc.append("checked" if checked else "unchecked")
            return f"No items found with {' and '.join(filter_desc)}"
        
        result = "Shopping list:\n"
        for item in sorted(filtered_items, key=lambda i: i["name"]):
            check_mark = "✓" if item.get("checked") else "○"
            quantity_str = f" ({item['quantity']})" if item.get("quantity") else ""
            category_str = f" [{item['category']}]" if item.get("category") else ""
            result += f"{check_mark} {item['name']}{quantity_str}{category_str}\n"
        
        return result.strip()
    
    async def add_item(
        self, 
        item: str, 
        quantity: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """Add item to shopping list."""
        item_key = item.lower()
        self.items[item_key] = {
            "name": item,
            "quantity": quantity,
            "category": category,
            "checked": False
        }
        
        quantity_str = f" ({quantity})" if quantity else ""
        category_str = f" in {category}" if category else ""
        return f"Added '{item}'{quantity_str} to shopping list{category_str}"
    
    async def remove_item(self, item: str) -> str:
        """Remove item from shopping list."""
        item_key = item.lower()
        if item_key not in self.items:
            return f"'{item}' not found in shopping list"
        
        self.items.pop(item_key)
        return f"Removed '{item}' from shopping list"
    
    async def check_item(self, item: str) -> str:
        """Mark item as checked."""
        item_key = item.lower()
        if item_key not in self.items:
            return f"'{item}' not found in shopping list"
        
        self.items[item_key]["checked"] = True
        return f"Checked off '{item}'"
    
    async def uncheck_item(self, item: str) -> str:
        """Mark item as unchecked."""
        item_key = item.lower()
        if item_key not in self.items:
            return f"'{item}' not found in shopping list"
        
        self.items[item_key]["checked"] = False
        return f"Unchecked '{item}'"
    
    async def clear_checked(self) -> str:
        """Remove all checked items."""
        checked_items = [k for k, v in self.items.items() if v.get("checked")]
        for key in checked_items:
            self.items.pop(key)
        
        return f"Removed {len(checked_items)} checked item(s)"
