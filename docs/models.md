# Two frozen encoder families

The sensor CNN takes [batch,128,6], transposes channels for temporal convolution and produces class logits plus a 64-dimensional embedding. It is shared by both systems. RGB frames are decoded into Farneback optical flow, summarized as spatial histograms and pooled into [batch,16,128]. A GRU or tiny Transformer processes this same HOF representation. No raw RGB Transformer or pretrained visual backbone is implied.

| Branch, 27-class head | Trainable parameters |
|---|---:|
| Sensor CNN | 29,883 |
| HOF-GRU | 39,259 |
| Tiny HOF-Transformer | 40,763 |
| Sensor + GRU | 69,142 |
| Sensor + Transformer | 70,646 |

Constructors and inference do not train, download data or load weights implicitly. Randomly initialized models demonstrate shapes only. A bundle identifies architecture, class ordering, preprocessing and normalization explicitly. Eight-class legacy weights cannot become full-class weights by changing labels.

For paired probabilities, arithmetic is `(sensor + video)/2`. Product uses float64 log probabilities with a 1e-12 floor, adds branch log scores, subtracts the row maximum, exponentiates and normalizes. Product adds no trainable parameters and need not universally sharpen confidence. Single-modality fallback preserves the available probabilities, regardless of requested fusion.
