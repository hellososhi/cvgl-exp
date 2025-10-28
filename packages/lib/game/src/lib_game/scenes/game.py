import cv2
import numpy as np
import pygame
from lib_pose import compute_similarity
from lib_pose.detect import PoseEstimator

from ..sequence import SceneInterface


class GameScene(SceneInterface):
    def __init__(self, manager=None):
        super().__init__(manager)
        self.cap = None
        self._estimator = None
        self.last_frame = None
        self.last_similarity = 0.0
        self._last_pose_data = None
        self._time_since_enter = 0.0
        self._transition_triggered = False
        self._auto_transition_delay = 5.0
        self._remaining_time = self._auto_transition_delay
        self._font = None
        self._timer_started = False

    def enter(self):
        """Open camera and prepare pose estimator and reference pose.

        The reference pose is generated randomly for the initial demo.
        """
        print("GameScene: enter")

        # open default camera
        if self.manager is None:
            raise RuntimeError("GameScene: no manager assigned")
        self.cap = self.manager.global_state.camera

        # long-lived estimator
        self._estimator = PoseEstimator(model_file="pose_landmarker_heavy.task")
        if self._estimator is None:
            raise RuntimeError("Could not create PoseEstimator")

        # generate a random reference pose to match against
        if self.manager is None:
            raise RuntimeError("GameScene: no manager assigned")
        self.reference_pose = self.manager.global_state.pose

        self.last_frame = None
        self.last_similarity = 0.0
        self._last_pose_data = None
        self._time_since_enter = 0.0
        self._transition_triggered = False
        self._remaining_time = self._auto_transition_delay
        if not pygame.font.get_init():
            pygame.font.init()
        self._font = pygame.font.SysFont(None, 48)
        self._timer_started = False

    def exit(self):
        print("GameScene: exit")
        if self._estimator is not None:
            self._estimator.close()
        if self.cap is not None:
            self.cap.release()
        self._estimator = None
        self.cap = None
        self._last_pose_data = None
        self._time_since_enter = 0.0
        self._transition_triggered = False
        self._remaining_time = self._auto_transition_delay
        self._font = None
        self._timer_started = False

    def update(self, dt: float) -> None:
        """Capture a frame, run pose estimation and compute similarity.

        This stores the last processed BGR frame in `self.last_frame` and
        the similarity score in `self.last_similarity` for rendering.
        """

        if self.cap is None or self._estimator is None:
            print("GameScene: update called before enter")
            return

        if not self._timer_started:
            self._timer_started = True
            self._time_since_enter = 0.0
        else:
            self._time_since_enter += dt

        ret, frame = self.cap.read()
        if not ret:
            print("GameScene: failed to read frame from camera")
            return

        # use the estimator to get PoseData
        pose = self._estimator.process_frame(frame)
        self._last_pose_data = pose
        self._flip_pose_data_x(pose)
        if self.manager is not None:
            state = self.manager.global_state
            state.query_pose = pose

        # compute similarity
        sim = 0.0
        if self.reference_pose:
            sim, _, _ = compute_similarity(self.reference_pose, pose)

        # mirror only for the user-facing preview surface
        display_frame = np.ascontiguousarray(frame[:, ::-1, :])
        print(pose)
        # display_frame.flags.writeable = True
        # draw_keypoints_on_frame(display_frame, pose)  # for debug
        # display_frame.flags.writeable = False
        self.last_frame = display_frame
        self.last_similarity = sim
        self._remaining_time = max(
            self._auto_transition_delay - self._time_since_enter, 0.0
        )

        if (
            not self._transition_triggered
            and self._time_since_enter >= self._auto_transition_delay
        ):
            self._transition_triggered = True
            if self.manager is not None:
                state = self.manager.global_state
                if self._last_pose_data is not None:
                    state.query_pose = self._last_pose_data
                self.manager.start("result")

    @staticmethod
    def _flip_pose_data_x(pose):
        if pose is None:
            return

        keypoints = getattr(pose, "keypoints", None)
        if keypoints is not None and keypoints.size > 0 and keypoints.shape[1] > 0:
            keypoints[:, 0] *= -1.0

        keypoints_world = getattr(pose, "keypoints_world", None)
        if (
            keypoints_world is not None
            and keypoints_world.size > 0
            and keypoints_world.shape[1] > 0
        ):
            keypoints_world[:, 0] *= -1.0

    def render(self, surface):
        """Convert the last processed frame to a pygame surface and blit it.

        The frame from OpenCV is BGR; convert to RGB and scale to surface size.
        """

        if surface is None:
            return

        if self.last_frame is None:
            # nothing to show yet
            return

        try:
            frame = self.last_frame
            # resize to surface size
            surf_w, surf_h = surface.get_size()
            frame_resized = cv2.resize(frame, (surf_w, surf_h))
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            # surface from buffer; ensure contiguous
            frame_rgb = np.ascontiguousarray(frame_rgb)
            pg_surf = pygame.image.frombuffer(
                frame_rgb.tobytes(), (surf_w, surf_h), "RGB"
            )
            surface.blit(pg_surf, (0, 0))

            if self._font is not None:
                countdown_text = self._font.render(
                    f"{self._remaining_time:.1f}s", True, (0, 255, 0)
                )
                surface.blit(countdown_text, (20, 60))
        except Exception as e:
            print("GameScene: render failed:", e)

    def handle_event(self, event) -> None:
        """Handle pygame events: SPACE returns to the start scene."""

        if event is None:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if self.manager is not None:
                    state = self.manager.global_state
                    state.query_pose = None
                    self.manager.start("start")
