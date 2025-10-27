from ..sequence import SceneInterface


class ResultScene(SceneInterface):
    def enter(self):
        print("ResultScene: enter")

    def exit(self):
        print("ResultScene: exit")

    def update(self, dt: float) -> None:
        pass

    def render(self, surface):
        print("ResultScene: render")

    def handle_event(self, event) -> None:
        pass
