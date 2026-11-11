# Method

Inputs are RGB and six IMU channels (acceleration and angular velocity), without
depth or skeletons. IMU channels are linearly resampled to 128 steps over relative
clip time and normalized with checkpoint training-fold means and standard deviations.
Two 1D convolutions (6→32, kernel 7; 32→64, kernel 5), adaptive pooling to four
steps, and a 256→64→8 head form the sensor branch (28,648 parameters).

Every video frame is resized to 160×120. Grayscale Farneback flow uses parameters
(0.5, 3, 9, 3, 5, 1.1, 0). Magnitudes strictly above 0.5 populate 16 temporal bins,
4×4 spatial cells and eight orientations. The whole histogram is divided by its
sum plus 1e-6. LayerNorm(128), a 64-unit GRU and an eight-way head use 38,024
parameters. No pretrained image backbone is used.

Both present: p = 0.5 p_sensor + 0.5 p_video. One present: use that branch exactly.
Neither present: insufficient_input without artifact access. Present zeros remain
observations. Known class IDs are 1,2,3,4,22,23,24,25: swipe left, swipe right, wave,
clap, jog, walk, sit to stand, stand to sit.

Choose one detector: D1 is mean cosine distance to five training-bank neighbors;
D2 is 1−max(p). Each availability pattern has its own saved threshold. A strict
score > threshold rejects as unknown. Otherwise C2 includes classes with 1−p≤q:
a singleton is accepted, all other cardinalities uncertain. Only accepted results
return action_id. Unknown results retain an explicitly diagnostic prediction set.

Temperature-scaled probabilities are reporting-only; decisions and C2 use raw p.
Fallback temperature fields are marked unrecommended. C2 uses descriptive pooled
crossfit calibration and supplies no formal new-person coverage guarantee.
