# PBS-Project

ffmpeg command to convert frames to video:

```bash
ffmpeg -framerate 24 -i frames/%06d.png -pix_fmt yuv420p output.mp4
```
