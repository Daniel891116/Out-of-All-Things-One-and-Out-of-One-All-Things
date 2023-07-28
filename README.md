# Out-of-All-Things-One-and-Out-of-One-All-Things
This project is inspired by the art work titled "Out of All Things One, and Out of One All Things" created by Petros Vrellis.

![DEMO](https://raw.githubusercontent.com/Daniel891116/Out-of-All-Things-One-and-Out-of-One-All-Things/main/src/variety.gif)
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project allows anyone to generate a 3 layers glass sheet image. If you stack them together, you will see the target image you choosed.

<!-- GETTING STARTED -->
## Getting Started

### Environment

This project can run on anaconda environent with Python version over 3.9
* Conda
  ```sh
  conda create -n [YourEnvName] python=3.9
  ```
### Prerequisites

Use this command to install required pip modules
* pip
  ```sh
  pip install -r requirements.txt
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. Choose the desired pattern or use the default patterns in the pattern directory

2. Choose your target image or use the default images in the src directory 

3. Start training!
    - place the target image path after --target
    - place the ssim score limit of the result to your target image after --ssim_limit

    ```sh
    python3 train.py --target src/cat1.jpg --ssim_limit 0.95
    ```
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


<!-- CONTACT -->
## Contact

Yao-ting, Huang yaotinghuang89@gmail.com

Project Link: [Out-of-All-Things-One-and-Out-of-One-All-Things](https://github.com/Daniel891116/Out-of-All-Things-One-and-Out-of-One-All-Things)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Petros Vrellis](http://artof01.com/vrellis/)
* [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)

<p align="right">(<a href="#readme-top">back to top</a>)</p>