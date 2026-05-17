# Tree species identification	

As the first step, the following data files are required to run the code in this repository:
- data/Z1_polygons.gpkg
- data/Z2_polygons.gpkg
- data/Z3_polygons.gpkg
- data/rgb_z1.tif
- data/rgb_z2.tif
- data/rgb_z3.tif

The `rgb_*.tif` files are masked RGB images of the three zones from the Quebec Trees dataset. The `Z*_polygons.gpkg` files are the corresponding vector files containing the tree species polygons for each zone.  


Next follow the instructions in `data/README.md` to prepare the data for training and testing the tree species identification model.  
  
Then follow the instructions in `rfdetrtrain/README.md` or `segformer/README.md` to train the tree species identification models.  
