from utils.user import User

class Map:

    def __init__(self) -> None:
        self.users: list[User] = []

    def add_user(self, user: User):
        self.users.append(user)