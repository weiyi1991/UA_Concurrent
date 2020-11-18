## Description
This is the UAlbany Concurrent Activity Detection Dataset (UA-CAD). ([Download link](http://169.226.117.122:8080/s/GzSE2zBnDjmkSmg))

This dataset contains 201 video sequences with concurrent activity annotations. We provide three data modalities as activity recognition input, namely 3D human skeleton joints, RGB video and depth maps.

The data and labels are stored in the following structure:

    seq_001
     |-----seq_001_c_s.mat    
     |-----seq_001_c.mp4
     |-----seq_001_d.mat
     |-----seq_001_label.txt
    seq_002
    ...

* seq_00x_c_s.mat: 3D skeleton sequence, 25 key joints are recored by Kinect V2.
* seq_00x_c.mp4: RGB video in 15 FPS.
* seq_00x_d.mat: depth map sequence
* seq_00x_label.txt: activity annotations, under each activity class is the [start frame, end frame]

Please use Matlab, Scipy or Octave to load `.mat` file. The skeleton sequence is stored in `body.Positions`.

All data modalities are recored at the same frame rate, 15 FPS.

Note that the depth map resolution is not consistent with RGB video.




## Citation

Please cite the following paper in your publications if UA-CAD helps your research.

    @inproceedings{DBLP:conf/aaai/WeiLFXCL20,
      author    = {Yi Wei and
                   Wenbo Li and
                   Yanbo Fan and
                   Linghan Xu and
                   Ming{-}Ching Chang and
                   Siwei Lyu},
      title     = {3D Single-Person Concurrent Activity Detection Using Stacked Relation
                   Network},
      booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
                   2020, The Thirty-Second Innovative Applications of Artificial Intelligence
                   Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
                   Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
                   February 7-12, 2020},
      pages     = {12329--12337},
      publisher = {{AAAI} Press},
      year      = {2020},
      url       = {https://aaai.org/ojs/index.php/AAAI/article/view/6917},
    }


## Contact
For more information or help, please get in touch with us via email.
* Yi Wei - ywei2@albany.edu
* Ming-Ching Chang - mchang2@albany.edu
