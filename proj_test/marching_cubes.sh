# python test.py
python marching_cube.py
ffmpeg -framerate 24 -i results/marching_cubes/%d.png -pix_fmt yuv420p output_marching.mp4 -y