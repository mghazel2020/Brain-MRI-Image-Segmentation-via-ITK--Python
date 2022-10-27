# Brain MRI Image Segmentation via ITK in Python

<img src="images/banner-002.jpg" width="1000" />

## 1. Objective

The objective of this project is to demonstrate the use SimpleITK to perform unimodal as well as multi-modal segmentation on a T1 and T2 MRI data.

## 2. Motivation

SimpleITK is a simplified programming interface to the algorithms and data structures of the Insight Segmentation and Registration Toolkit (ITK). It supports bindings for multiple programming languages including Python.  Combining SimpleITK's Python binding with the Jupyter notebook web application creates an environment which facilitates collaborative development of biomedical image analysis workflows.


In particular, we shall implement the following two types of segmentation using SimpleITK Python::
    * Uni-modal segmentation of two different modality images (T1 and T1), independently.
    * Multi-model segmentation by combing the T1 and T2 images to achieve more accurate segmentation.
    
The applied segmentation algorithms are semi-automated in they sense
    * The user provides an initial set of seed points of the region of the interest
    * A region-growing algorithm is then applied in order to segment the image based on the provided seeds.

We shall demonstrate the various steps involved in this task as we develop the code.

## 3. Data

We make use of the following data set:

    * Source: Retrospective Image Registration Evaluation Project
    * The RIRE Project provides patient datasets acquired with different imaging modalities, e.g., MR, CT, PET
    * These data sets are widely used in evaluation of different image registration and segmentation techniques
    * Link: https://www.insight-journal.org/rire/download_data.php
    * Used test data:
        * Patient: 101
        * Modalities: T1 and T2.

## 4. Development

In this section, we shall demonstrate the use SimpleITK to perform unimodal as well as multi-modal segmentation from the T1 and T2 MRI data above.

* Author: Mohsen Ghazel
* Date: March 26th, 2021
* Project: Human brain MRI imaging segmentation using SimpleITK:

The objective of this project is to demonstrate how to perform semi-automated segmentation of multi-modal MRI brain images using SimpleITK:

* In particular, the following two types of segmentation:
    * Uni-modal segmentation of two different modality images (T1 and T1), independently.
    * Multi-model segmentation by combing the T1 and T2 images to achieve more accurate segmentation.
* The applied segmentation algorithms are semi-automated in they sense:
    * The user provides an initial set of seed points of the region of the interest
    * A region-growing algorithm is then applied in order to segment the image based on the provided seeds.
* We shall demonstrate the various steps involved in this task as we develop the code.

### 4.1. Step 1: Python imports and global variables


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Python imports and environment setup</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># I/O os</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># Opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># Numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np

<span style="color:#696969; "># Matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>image <span style="color:#800000; font-weight:bold; ">as</span> mpimg

<span style="color:#696969; "># date-time to show date and time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># SimpleITK library</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># for installation run one of the the following </span>
<span style="color:#696969; "># commands:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># &gt;&gt; conda install -c simpleitk simpleitk</span>
<span style="color:#696969; "># &gt;&gt; pip install SimpleITK</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">import</span> SimpleITK

<span style="color:#696969; "># set: %matplotlib inline so that matplotlib graphs </span>
<span style="color:#696969; "># will be included in your notebook, next to the code</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#696969; "># Testing the OpenCV version</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV : "</span><span style="color:#808030; ">,</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span>

<span style="color:#696969; "># Testinng the numpy version</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy : "</span><span style="color:#808030; ">,</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span>

<span style="color:#696969; "># Testinng the SimpleITK version</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"SimpleITK : "</span><span style="color:#808030; ">,</span>SimpleITK<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span>

OpenCV <span style="color:#808030; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#808030; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span>
SimpleITK <span style="color:#808030; ">:</span>  <span style="color:#008000; ">2.0</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span>
</pre>

### 4.2. Step 2: Implement the helper functions:

* These are helper functions used for processing and visualizing the data

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">"""Display 2D SimpleITK image with a grayscale colormap and accompanying axes</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;Parameters:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;img (int): input image</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;title (string): figure title</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;margin (int): figure margin</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;dpi (int): figure dpi resolution</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;None</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;"""</span>
<span style="color:#800000; font-weight:bold; ">def</span> itk_visualize<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> title<span style="color:#808030; ">=</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> margin<span style="color:#808030; ">=</span><span style="color:#008000; ">0.0</span><span style="color:#808030; ">,</span> dpi<span style="color:#808030; ">=</span><span style="color:#008c00; ">40</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    nda <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>GetArrayFromImage<span style="color:#808030; ">(</span>img<span style="color:#808030; ">)</span>
    <span style="color:#696969; ">#spacing = img.GetSpacing()</span>
    figsize <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span> <span style="color:#44aadd; ">+</span> margin<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">*</span> nda<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">/</span> dpi<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span> <span style="color:#44aadd; ">+</span> margin<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">*</span> nda<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">/</span> dpi
    <span style="color:#696969; ">#extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)</span>
    extent <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> nda<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> nda<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span>
    fig <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span>figsize<span style="color:#808030; ">,</span> dpi<span style="color:#808030; ">=</span>dpi<span style="color:#808030; ">)</span>
    ax <span style="color:#808030; ">=</span> fig<span style="color:#808030; ">.</span>add_axes<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>margin<span style="color:#808030; ">,</span> margin<span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">*</span>margin<span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">*</span>margin<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display is gray color-map</span>
    plt<span style="color:#808030; ">.</span>set_cmap<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"gray"</span><span style="color:#808030; ">)</span>
    ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>nda<span style="color:#808030; ">,</span>extent<span style="color:#808030; ">=</span>extent<span style="color:#808030; ">,</span>interpolation<span style="color:#808030; ">=</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># add the figure titke if provided</span>
    <span style="color:#800000; font-weight:bold; ">if</span> title<span style="color:#808030; ">:</span>
        plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span>title<span style="color:#808030; ">)</span>
    <span style="color:#696969; "># show figure</span>
    plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">"""Tile multiple images together into a vector:</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;Parameters:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;lstImgs (lstImgs): list input image</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;None</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;"""</span>
<span style="color:#800000; font-weight:bold; ">def</span> itk_tile_images_vector<span style="color:#808030; ">(</span>lstImgs<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    lstImgToCompose <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">]</span>
    <span style="color:#800000; font-weight:bold; ">for</span> idxComp <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>lstImgs<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">.</span>GetNumberOfComponentsPerPixel<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        lstImgToTile <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">]</span>
        <span style="color:#800000; font-weight:bold; ">for</span> img <span style="color:#800000; font-weight:bold; ">in</span> lstImgs<span style="color:#808030; ">:</span>
            lstImgToTile<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>VectorIndexSelectionCast<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> idxComp<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
        lstImgToCompose<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>Tile<span style="color:#808030; ">(</span>lstImgToTile<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>lstImgs<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    itk_visualize<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>Compose<span style="color:#808030; ">(</span>lstImgToCompose<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#0000e6; ">"Uni-model Gray-Matter Segmentation"</span><span style="color:#808030; ">)</span>
</pre>


### 4.3. Step 3: Read the input images:

* We make use of the following data set:
    * Source: Retrospective Image Registration Evaluation Project
    * The RIRE Project provides patient datasets acquired with different imaging modalities, e.g., MR, CT, PET
    * These data sets are widely used in evaluation of different image registration and segmentation techniques
    * Link: https://www.insight-journal.org/rire/download_data.php
    * Used test data:
      * Patient: 101
      * Modalities: T1 and T2.

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Paths to the .mhd T1 and T2 test images files</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># T1-image file</span>
filenameT1 <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"./resources/RIRE/patient_101/mr_T1/patient_101_mr_T1.mhd"</span>
<span style="color:#696969; "># T2-image file</span>
filenameT2 <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"./resources/RIRE/patient_101/mr_T2/patient_101_mr_T2.mhd"</span>
</pre>

#### 4.3.1. Specify the used slice:

* The T1 and T2 modalities are 3D data sets:
    * We shall use a particular slice in order to extract 2D images
    * We elected to use slice # 25, but one may choose any other valid slice index.
    

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Slice index to visualize with the itk_visualize helper function</span>
idxSlice <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span>
</pre>

4.3.2. Class of interest:

* As mentioned earlier, we aim to segment the brain MRI T1 and T2 slice images:
  * The brain is mainly composed of two components:
    * Gray matter
    * White matter
  * We need to assign integer-labels to these two classes.
    * These need to be different integers but their values themselves are not important
      * We assign: labelGrayMatter = 1
      * We assign: labelWhiteMatter = 2


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Assign an integer label to assign to the gray matter</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
labelGrayMatter <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Assign an integer label to assign to the white matter</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
labelGrayMatter <span style="color:#808030; ">=</span> <span style="color:#008c00; ">2</span>
</pre>


### 4.4. Step 4: Region-growing seeds:

* The applied segmentation algorithms are semi-automated in they sense:
    * The user provides an initial set of seed points of the region of the interest
    * A region-growing algorithm is then applied in order to segment the image based on the provided seeds.
 * Thus, we need to provide a small set of initial seeds belonging to the class of interest (Gray-Matter)
    * These points are selected from the T2 slice image
    * These seeds are specific to the input T2 slice image

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Specify the region growing seed points for the input </span>
<span style="color:#696969; "># image</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Patient: 101</span>
<span style="color:#696969; "># Slide: 25</span>
<span style="color:#696969; "># Modality: T2</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># array of seeds </span>
lstSeeds <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">185</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">130</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
            <span style="color:#808030; ">(</span><span style="color:#008c00; ">110</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">165</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
            <span style="color:#808030; ">(</span><span style="color:#008c00; ">165</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">175</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
            <span style="color:#808030; ">(</span><span style="color:#008c00; ">145</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">50</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
            <span style="color:#808030; ">(</span><span style="color:#008c00; ">125</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">115</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">)</span><span style="color:#808030; ">]</span>
</pre>

### 4.5. Step 5: Read and visualize the data:

* Read and visualize the selected slice number from the T1 and T2 data modalities


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Paths to the input data .mhd files</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># T1 data file</span>
filenameT1 <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"./resources/RIRE/patient_101/mr_T1/patient_101_mr_T1.mhd"</span>
<span style="color:#696969; "># T2 data file</span>
filenameT2 <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">"./resources/RIRE/patient_101/mr_T2/patient_101_mr_T2.mhd"</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># read the data</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># T1 data</span>
imgT1Original <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>ReadImage<span style="color:#808030; ">(</span>filenameT1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># T2 data</span>
imgT2Original <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>ReadImage<span style="color:#808030; ">(</span>filenameT2<span style="color:#808030; ">)</span>

<span style="color:#696969; "># visualize the T1 and T2 slices side by side (idxSlice = 25)</span>
itk_visualize<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>Tile<span style="color:#808030; ">(</span>imgT1Original<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span>
                         imgT2Original<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span>
                         <span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#0000e6; ">"Original data"</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/original-slice-25-T1-T2-images.jpg" width="1000" />

### 4.6. Step 6: Image pre-processing:

  * As we can see from the above figure, the original image data exhibits quite a bit of noise which is very typical of MRI datasets.
  * However, since we will be applying region-growing and thresholding segmentation algorithms we need a smoother, more homogeneous pixel distribution.
  * Thus, before we start the segmentation, we need smoothen the images in order to reduce the noise side-effects on the segmentation results.
  

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Apply smoothing on the T1 modality slide image</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgT1Smooth <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>CurvatureFlow<span style="color:#808030; ">(</span>image1<span style="color:#808030; ">=</span>imgT1Original<span style="color:#808030; ">,</span>
                                      timeStep<span style="color:#808030; ">=</span><span style="color:#008000; ">0.125</span><span style="color:#808030; ">,</span>
                                      numberOfIterations<span style="color:#808030; ">=</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Apply smoothing on the T2 modality slide image</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgT2Smooth <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>CurvatureFlow<span style="color:#808030; ">(</span>image1<span style="color:#808030; ">=</span>imgT2Original<span style="color:#808030; ">,</span>
                                      timeStep<span style="color:#808030; ">=</span><span style="color:#008000; ">0.125</span><span style="color:#808030; ">,</span>
                                      numberOfIterations<span style="color:#808030; ">=</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-----------------------------------------------</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
itk_visualize<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>Tile<span style="color:#808030; ">(</span>imgT1Smooth<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> 
                         imgT2Smooth<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> 
                         <span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#0000e6; ">"After smoothing"</span><span style="color:#808030; ">)</span>
</pre>


<img src="images/after-smoothing-T1-T2-slice-25.png" width="1000" />

### 4.7. Step 7: Initial seeds visualization:

* Overlay the selected initial gray-matter seeds on the smoothed T2 slice image.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Loop through the seeds defined in lstSeeds and set those </span>
<span style="color:#696969; "># pixels to a high value so they will stand out and allow </span>
<span style="color:#696969; "># us to see them through itk_visualize. </span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgSeeds <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>Image<span style="color:#808030; ">(</span>imgT2Smooth<span style="color:#808030; ">)</span>

<span style="color:#800000; font-weight:bold; ">for</span> s <span style="color:#800000; font-weight:bold; ">in</span> lstSeeds<span style="color:#808030; ">:</span>
    imgSeeds<span style="color:#808030; ">[</span>s<span style="color:#808030; ">]</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">10000</span>

itk_visualize<span style="color:#808030; ">(</span>imgSeeds<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> <span style="color:#0000e6; ">"Selected Region-growing Seeds"</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/seed-points-overlay.png" width="1000" />

### 4.8. Step 8: Uni-Modal Segmentation:

* First, starting from the selected intial seed points, we apply the region-growing algorithm to each of the two images, separately:
    * The smoothed T1-slice image imgT1Smooth
    * The smoothed T2-slice image imgT2Smooth
    * This perform uni-model image segmentation as the 2 images are segmented independently.
    

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Starting from the selected initial seed points, </span>
<span style="color:#696969; "># we apply the region-growing # on the smoothed </span>
<span style="color:#696969; "># T1-slice image imgT1Smooth</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgGrayMatterT1 <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>ConfidenceConnected<span style="color:#808030; ">(</span>image1<span style="color:#808030; ">=</span>imgT1Smooth<span style="color:#808030; ">,</span> 
                                                seedList<span style="color:#808030; ">=</span>lstSeeds<span style="color:#808030; ">,</span>
                                                numberOfIterations<span style="color:#808030; ">=</span><span style="color:#008c00; ">7</span><span style="color:#808030; ">,</span>
                                                multiplier<span style="color:#808030; ">=</span><span style="color:#008000; ">1.0</span><span style="color:#808030; ">,</span>
                                                replaceValue<span style="color:#808030; ">=</span>labelGrayMatter<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Starting from the selected intial seed points, </span>
<span style="color:#696969; "># we apply the region-growing on the smoothed </span>
<span style="color:#696969; "># T2-slice image imgT2Smooth</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgGrayMatterT2 <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>ConfidenceConnected<span style="color:#808030; ">(</span>image1<span style="color:#808030; ">=</span>imgT2Smooth<span style="color:#808030; ">,</span> 
                                                seedList<span style="color:#808030; ">=</span>lstSeeds<span style="color:#808030; ">,</span>
                                                numberOfIterations<span style="color:#808030; ">=</span><span style="color:#008c00; ">7</span><span style="color:#808030; ">,</span>
                                                multiplier<span style="color:#808030; ">=</span><span style="color:#008000; ">1.5</span><span style="color:#808030; ">,</span>
                                                replaceValue<span style="color:#808030; ">=</span>labelGrayMatter<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># create a rescaled integer version of imgT1Smooth </span>
<span style="color:#696969; "># named imgT1SmoothInt:</span>
<span style="color:#696969; "># - Needed in order to create label overlays</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgT1SmoothInt <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>Cast<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>RescaleIntensity<span style="color:#808030; ">(</span>imgT1Smooth<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                                imgGrayMatterT1<span style="color:#808030; ">.</span>GetPixelID<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># create a rescaled integer version of imgT2Smooth </span>
<span style="color:#696969; "># named imgT2SmoothInt:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># - Needed in order to create label overlays</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgT2SmoothInt <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>Cast<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>RescaleIntensity<span style="color:#808030; ">(</span>imgT2Smooth<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                                imgGrayMatterT2<span style="color:#808030; ">.</span>GetPixelID<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># visualize the uni-model segmentation results for the 2 images</span>
itk_tile_images_vector<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>SimpleITK<span style="color:#808030; ">.</span>LabelOverlay<span style="color:#808030; ">(</span>imgT1SmoothInt<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span>idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> 
                                      imgGrayMatterT1<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span>idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
               SimpleITK<span style="color:#808030; ">.</span>LabelOverlay<span style="color:#808030; ">(</span>imgT2SmoothInt<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span>idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> 
                                     imgGrayMatterT2<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span>idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/uni-modal-segmentation.png" width="1000" />

### 4.9. Step 9: Perform multi-modal segmentation:

* Multi-modal segmentation gives us multiple views of the same anatomies:

    * In our case by combining the T1 and T2 images we get two significantly different views of the gray matter and areas that were not connected in one of the images may be connected in the other.
    * In addition, specious connections appearing in one image due to excessive noise or artifacts will most likely not appear in the other.
    (* This allows us to perform the segmentation having more information in our arsenal and achieving the same or better results with more stringent criteria and far fewer iterations.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Firstly, we begin by combining the two images: </span>
<span style="color:#696969; ">#  - In order to do that we simply use the ComposeImageFilter </span>
<span style="color:#696969; ">#  - The result of the composition is the imgComp image.</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgComp <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>Compose<span style="color:#808030; ">(</span>imgT1Smooth<span style="color:#808030; ">,</span> imgT2Smooth<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Perorm the segmentation, using the VectorConfidenceConnectedImageFilter </span>
<span style="color:#696969; "># class instead of the ConfidenceConnectedImageFilter, </span>
<span style="color:#696969; "># used for the uni-modal segmentation</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
imgGrayMatterComp <span style="color:#808030; ">=</span> SimpleITK<span style="color:#808030; ">.</span>VectorConfidenceConnected<span style="color:#808030; ">(</span>image1<span style="color:#808030; ">=</span>imgComp<span style="color:#808030; ">,</span> 
                                               seedList<span style="color:#808030; ">=</span>lstSeeds<span style="color:#808030; ">,</span>
                                               numberOfIterations<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span>
                                               multiplier<span style="color:#808030; ">=</span><span style="color:#008000; ">0.1</span><span style="color:#808030; ">,</span>
                                               replaceValue<span style="color:#808030; ">=</span>labelGrayMatter<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># display the multi-model image segmentation results</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
itk_visualize<span style="color:#808030; ">(</span>SimpleITK<span style="color:#808030; ">.</span>LabelOverlay<span style="color:#808030; ">(</span>imgT2SmoothInt<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span>idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> 
                                 imgGrayMatterComp<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span>idxSlice<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#0000e6; ">"Multi-model Gray-Matter Segmentation"</span><span style="color:#808030; ">)</span>
</pre>


<img src="images/multi-modal-segmentation.png" width="1000" />

#### 4.9.1. Observations:

* The multi-modal segmentation results appear much better than the unimodal results overlaid on T2
* Not only did we get rid of the skin and fat parts that segmentation on the T1 image gave us, but we also segmented pretty much all gray matter
* Of course the segmentation is not perfect as multiple areas of white matter are also incorrectly segmented as part of the gray matter
* Given our simple approach, we achieved reasonably good gray-matter segmentation results using the SimpleITK Python API.

### 4.10> Step 10: Save the multi-model segmentation MHD output:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># save the final multi-model segmentation MHD output </span>
<span style="color:#696969; "># imgGrayMatterCom</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
SimpleITK<span style="color:#808030; ">.</span>WriteImage<span style="color:#808030; ">(</span>imgGrayMatterComp<span style="color:#808030; ">,</span> <span style="color:#0000e6; ">"GrayMatter.mhd"</span><span style="color:#808030; ">)</span>
</pre>


### 4.11. Step 11: Display a final message after successful execution


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># display a final message</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
</pre>

## 5. Analysis

* In view of the presented results, we make the following observations:
  * Not only did we get rid of the skin and fat parts that segmentation on the T1 image gave us, but we also segmented pretty much all gray matter
  * Of course the segmentation is not perfect as multiple areas of white matter are also incorrectly segmented as part of the gray matter
  * Given our simple approach, we achieved reasonably good gray-matter segmentation results using the SimpleITK Python API.
  * The multi-modal segmentation results appear much better than the unimodal results overlaid on T2.


## 6. Future Work

* We propose to investigate the following related tasks:
    * To explore the sensitivity of the segmentation results to the initially selected seeds from the region of interest (gray-matter)
    * To explore segmenting diseased tissues in brain MRI, such multiple sclerosis lesions.
    * To explore processing the full T1 and T2 3D data sets instead of just individual slices.


## 7. References

1. SimpleITK. https://simpleitk.org/
2. Somada141. Multi-Modal Image Segmentation with Python & SimpleITK. https://pyscience.wordpress.com/2014/11/02/multi-modal-image-segmentation-with-python-simpleitk/
3. Somada141. Image Segmentation with Python and SimpleITK. https://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/








