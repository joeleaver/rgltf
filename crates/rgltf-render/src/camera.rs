//! Orbit camera: yaw/pitch around a target, mouse-driven.

use glam::{Mat4, Vec3};

/// A right-handed orbit camera producing a wgpu-convention (z ∈ [0,1]) view-projection.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Horizontal angle around the target (radians).
    pub yaw: f32,
    /// Vertical angle (radians), clamped away from the poles.
    pub pitch: f32,
    /// Distance from the target.
    pub distance: f32,
    /// Look-at target in world space.
    pub target: Vec3,
    /// Vertical field of view (radians).
    pub fovy: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// Bounding radius of the framed content; used to derive clip planes
    /// dynamically each frame so zooming never clips or z-fights.
    pub radius: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.45,
            distance: 3.5,
            target: Vec3::ZERO,
            fovy: 45f32.to_radians(),
            aspect: if aspect.is_finite() && aspect > 0.0 { aspect } else { 1.0 },
            radius: 1.0,
        }
    }

    /// Near/far planes derived from the current view distance and content radius.
    ///
    /// `near` is a small fraction of the view distance rather than hugging the
    /// rest-pose content sphere — animation (and multi-object scenes) can move
    /// geometry closer to the eye than `distance - radius`, which would otherwise
    /// near-clip it. `far` is generous for the same reason. `Depth32Float` has ample
    /// precision for the resulting ratio, and shrinking `near` with `distance` means
    /// zooming/dollying in never clips.
    fn clip_planes(&self) -> (f32, f32) {
        let r = self.radius.max(1e-4);
        let far = (self.distance + r * 3.0).max(r * 1e-2);
        let near = (self.distance * 0.02).clamp(far * 1e-4, far * 0.5);
        (near, far.max(near * 1.0001))
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        if aspect.is_finite() && aspect > 0.0 {
            self.aspect = aspect;
        }
    }

    /// World-space eye position derived from yaw/pitch/distance.
    pub fn eye(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        let dir = Vec3::new(cp * sy, sp, cp * cy);
        self.target + dir * self.distance
    }

    /// Combined view-projection matrix (right-handed, wgpu depth range).
    pub fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let (near, far) = self.clip_planes();
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, near, far);
        proj * view
    }

    /// Orbit by pixel deltas (dx horizontal, dy vertical). Dragging down tilts the
    /// view down (camera rises), matching common 3D-viewer convention.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const SENS: f32 = 0.008;
        self.yaw -= dx * SENS;
        self.pitch = (self.pitch + dy * SENS).clamp(-1.5, 1.5);
    }

    /// Dolly toward/away from the target. `delta` is wheel delta_y; scrolling up
    /// (wheel forward) zooms in, matching common convention.
    pub fn zoom(&mut self, delta: f32) {
        let factor = (1.0 - delta * 0.001).clamp(0.5, 1.5);
        self.distance = (self.distance * factor).clamp(1e-4, 1e6);
    }

    /// Frame a bounding sphere (center + radius): reset to a pleasant 3/4 view and
    /// pull back far enough to fit the sphere in the vertical FOV. Also rescales the
    /// clip planes to the model size so near/far don't clip or z-fight.
    pub fn frame(&mut self, center: Vec3, radius: f32) {
        let radius = radius.max(1e-4);
        self.target = center;
        self.radius = radius;
        self.yaw = 0.7;
        self.pitch = 0.45;
        // Fit the sphere in the (smaller) vertical half-angle, with margin.
        let half = (self.fovy * 0.5).min(self.horizontal_half_fov());
        self.distance = (radius / half.sin()) * 1.25;
    }

    fn horizontal_half_fov(&self) -> f32 {
        // Vertical fov → horizontal, given aspect; used so wide viewports still fit.
        let vt = (self.fovy * 0.5).tan();
        (vt * self.aspect.max(1e-3)).atan()
    }

    /// Pull back far enough to fit the framed sphere in the FOV (with margin).
    fn fit_distance(&self) -> f32 {
        let half = (self.fovy * 0.5).min(self.horizontal_half_fov());
        (self.radius / half.sin().max(1e-4)) * 1.25
    }

    /// Snap to a view angle (keeps the current target + content radius) and re-fit the
    /// distance so the model fills the frame. Used by the camera-preset buttons.
    pub fn set_view(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-1.5, 1.5);
        self.distance = self.fit_distance();
    }

    /// Restore the default 3/4 framing of the current content (the "reset" button).
    pub fn refit(&mut self) {
        self.set_view(0.7, 0.45);
    }
}
