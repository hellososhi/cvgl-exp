import cv2
import numpy as np
import pygame
from lib_pose import compute_similarity
from lib_pose.detect import PoseEstimator
from lib_pose.util_2d import draw_keypoints_on_frame, draw_similarity_on_frame

from ..sequence import SceneInterface


class GameScene(SceneInterface):
    def __init__(self, manager=None):
        super().__init__(manager)
        self.cap = None
        self._estimator = None
        self.last_frame = None
        self.last_similarity = 0.0
        self._last_pose_data = None

    def enter(self):
        """Open camera and prepare pose estimator and reference pose.

        The reference pose is generated randomly for the initial demo.
        """
        print("GameScene: enter")

        # open default camera
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # long-lived estimator
        self._estimator = PoseEstimator()
        if self._estimator is None:
            raise RuntimeError("Could not create PoseEstimator")

        # generate a random reference pose to match against
        if self.manager is None:
            raise RuntimeError("GameScene: no manager assigned")
        self.reference_pose = self.manager.global_state.pose

        self.last_frame = None
        self.last_similarity = 0.0
        self._last_pose_data = None

    def exit(self):
        print("GameScene: exit")
        if self._estimator is not None:
            self._estimator.close()
        if self.cap is not None:
            self.cap.release()
        self._estimator = None
        self.cap = None
        self._last_pose_data = None

    def update(self, dt: float) -> None:
        """Capture a frame, run pose estimation and compute similarity.

        This stores the last processed BGR frame in `self.last_frame` and
        the similarity score in `self.last_similarity` for rendering.
        """

        if self.cap is None or self._estimator is None:
            print("GameScene: update called before enter")
            return

        ret, frame = self.cap.read()
        if not ret:
            print("GameScene: failed to read frame from camera")
            return

        # use the estimator to get PoseData
        pose = self._estimator.process_frame(frame)
        self._last_pose_data = pose
        if self.manager is not None:
            state = self.manager.global_state
            state.query_pose = pose

        # compute similarity
        sim = 0.0
        if self.reference_pose:
            sim, _ = compute_similarity(self.reference_pose, pose)

        # draw overlays directly onto a copy of the frame (BGR)
        vis_frame = frame.copy()
        draw_keypoints_on_frame(vis_frame, pose)
        draw_similarity_on_frame(vis_frame, sim)

        self.last_frame = vis_frame
        self.last_similarity = sim
        if self.manager is not None:
            state = self.manager.global_state
            state.similarity = sim

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
        except Exception as e:
            print("GameScene: render failed:", e)

    def handle_event(self, event) -> None:
        """Handle pygame events: ESC quits to result scene, SPACE returns to start."""

        if event is None:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # go to result scene
                if self.manager is not None:
                    state = self.manager.global_state
                    if self._last_pose_data is not None:
                        state.query_pose = self._last_pose_data
                    self.manager.start("result")
            elif event.key == pygame.K_SPACE:
                if self.manager is not None:
                    state = self.manager.global_state
                    state.query_pose = None
                    state.similarity = 0.0
                    state.transformed_pose = None
                    self.manager.start("start")
