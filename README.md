<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!--
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  -->
  <h3 align="center">IPMandCamCalib</h3>

  <p align="center">
    This is a repository for high performance real time camera calibration and performing inverse perspective mapping at the same time.
    <!--
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    -->
    <br />
    <a href="https://github.com/butnaruteodor/IPMandCamCalib">View Demo</a>
    ·
    <a href="https://github.com/butnaruteodor/IPMandCamCalib/issues">Report Bug</a>
    ·
    <a href="https://github.com/butnaruteodor/IPMandCamCalib/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<!--
[![Product Name Screen Shot][product-screenshot]](https://example.com)
-->
I've had problems with finding a performant algorithm which calibrates the camera and performs inverse perspective mapping to get the bird eye view variant of an image to use in my self driving car project so I made this repository to help other people that want a high performance camera calibration and/or ipm algorithm.

The IPM code is heavily inspired from [IPM-master][IPM-master-url] by JamesLiao so huge thanks for his work!

The code was built primarily to run on jetson nano although it should work on any cuda capable device.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!--### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

<p align="right">(<a href="#readme-top">back to top</a>)</p>-->



<!-- GETTING STARTED -->
## Getting Started

First you should get started by trying out the demo

### Prerequisites

Here are all the dependencies you will need:
* cuda 10.1 or 10.2
  
  Follow the official documentation from NVIDIA [CUDA][CUDA-url]
* gcc 8
  ```sh
  sudo apt-get install gcc-8 g++-8
  ```
* cmake
  ```sh
  sudo apt-get install cmake
  ```
* eigen
  ```sh
  sudo apt install libeigen3-dev
  ```
* opencv (needed only for demo)
  
  Follow the official documentation [OpenCV][OpenCV-url]

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/butnaruteodor/IPMandCamCalib.git
   ```
2. Build the project
   ```sh
   chmod +x build.sh && ./build.sh
   ```
3. Run the demo
   ```sh
   ./IPM
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Add documentation
- [ ] Add usage examples
- [ ] Broader compatibility

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
<!--## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- CONTACT -->
## Contact

Butnaru Teodor - butnaruteodor@gmail.com

Project Link: [https://github.com/butnaruteodor/IPMandCamCalib](https://github.com/butnaruteodor/IPMandCamCalib)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [IPM-master](https://github.com/JamesLiao714/IPM-master)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/butnaruteodor/IPMandCamCalib/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/butnaruteodor/IPMandCamCalib/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/butnaruteodor/IPMandCamCalib/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/butnaruteodor/IPMandCamCalib/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/teodor-butnaru/
[IPM-master-url]: https://github.com/JamesLiao714/IPM-master
[CUDA-url]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
[OpenCV-url]: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
