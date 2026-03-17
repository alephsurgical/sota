# Subtask Labels

The SOTA dataset contains surgical robotics data for the **anastomosis** task - suturing blood vessels together. Each episode is annotated with frame-level subtask labels describing the surgical procedure.

## Label Vocabulary

The anastomosis task is decomposed into 17 distinct subtasks, plus 2 special labels:

| Index | Label | Description |
|-------|-------|-------------|
| 0 | `grasp needle body with driver` | Pick up the needle using the needle driver tool |
| 1 | `align needle to vessel axis with forceps` | Position the needle at the correct angle for insertion |
| 2 | `insert forceps into right vein opening` | Place forceps into the right vessel to spread it open |
| 3 | `spread right vein with forceps` | Open the forceps to widen the vessel opening |
| 4 | `insert needle through right vein with driver` | Pass the needle through the right vessel wall |
| 5 | `pull forceps out of right vein` | Remove forceps from the vessel |
| 6 | `grasp left vein with forceps` | Hold the left vessel with forceps |
| 7 | `insert needle through left vein with driver` | Pass the needle through the left vessel wall |
| 8 | `release needle with driver` | Let go of the needle |
| 9 | `grasp needle tip with driver` | Re-grasp the needle by its tip |
| 10 | `pull needle through left vein with driver` | Pull the needle completely through |
| 11 | `pull suture up with driver until short tail remains` | Tighten the suture, leaving a small tail |
| 12 | `position forceps as post for loop, open` | Set up forceps to create a knot loop |
| 13 | `wrap suture around forceps` | Loop the thread around the forceps |
| 14 | `grasp suture tail with forceps` | Grab the short end of the suture |
| 15 | `pull forceps and driver in opposite directions until knot tight` | Tighten the surgical knot |
| 16 | `place needle on workspace with driver` | Put down the needle |
| 17 | `failure` | Episode contains a failure or error |
| 18 | `unused` | Frames before/after the main task (setup, cleanup) |

## Annotation Format

Subtask annotations are stored in `meta/subtask_annotations.json`:

```json
{
  "version": "1.0",
  "episodes": {
    "0": {
      "segments": [
        {
          "start_frame": 0,
          "end_frame": 15,
          "label": "unused",
          "source": "manual",
          "confidence": null
        },
        {
          "start_frame": 15,
          "end_frame": 160,
          "label": "grasp needle body with driver",
          "source": "manual",
          "confidence": null
        }
      ],
      "annotated_at": "2026-03-15T03:24:08.954393Z"
    }
  }
}
```

### Fields

- `start_frame` / `end_frame`: Frame range within the episode (0-indexed, exclusive end)
- `label`: Subtask label from the vocabulary
- `source`: `"manual"` (human labeled) or `"model"` (predicted)
- `confidence`: Model confidence score (0-1) for predictions, `null` for manual labels

## Using Annotations

```python
from sota_data import get_subtask_annotations, load_subtask_labels

# Load label vocabulary
labels = load_subtask_labels("twoarm-anastomosis-real-dagger-200")
print(labels)

# Load annotations for a session
annotations = get_subtask_annotations(
    "twoarm-anastomosis-real-dagger-200/20260311_170113"
)

# Get label for a specific frame
episode_idx = 0
frame_offset = 100

for seg in annotations["episodes"]["0"]["segments"]:
    if seg["start_frame"] <= frame_offset < seg["end_frame"]:
        print(f"Frame {frame_offset}: {seg['label']}")
        break
```

## Subtask Index in Data

Each frame in the parquet data files includes a `subtask_index` column that maps to the label vocabulary. This allows efficient filtering:

```python
from sota_data import load_dataset

dataset = load_dataset("twoarm-anastomosis-real-dagger-200/20260311_170113")
frame = dataset[100]

subtask_idx = frame["subtask_index"]
labels = load_subtask_labels("twoarm-anastomosis-real-dagger-200")
print(f"Subtask: {labels[subtask_idx]}")
```
