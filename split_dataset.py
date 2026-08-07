
import splitfolders

splitfolders.ratio(
    "data/corrosion detect",
    output="Corrosion_Split",
    seed=42,
    ratio=(0.8, 0.1, 0.1)
)