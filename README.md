# debug_argoverse2
It is helpful to understand how the Argover v2 is structured, it only projects the 3d bounding boxes into 2D bounding boxes or convex hull as well, this is done without using the [argoverse-api](https://github.com/argoverse/argoverse-api) because I had problems with the installation.



It has been tested only on front camera and on the first chunk of the Argoverse V2 dataset (`train-000`), the files are large, this one is circa ~`56GB`.



## Projecting 2D Bounding Boxes

![](images/argoverseV2_amodal_annotation.jpg)

## Projecting 2D Convex Hulls (Bounding Polygons)

![](images/315972616049927208_hulls.jpg)

## Checking intersection among Bounding Polygons.

![](images/argover2_area_overlapping.png)

## Requirements in Python

- pandas
- matplolib
- PIL
- numpy
- scipy
