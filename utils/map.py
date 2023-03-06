from utils.user import User
from pathlib import Path

class Map:

    def __init__(self, path: Path) -> None:
        self.users: list[User] = []
        self.path = path

    def add_user(self, user: User):
        self.users.append(user)