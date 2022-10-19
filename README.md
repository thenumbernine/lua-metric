[![Donate via Stripe](https://img.shields.io/badge/Donate-Stripe-green.svg)](https://buy.stripe.com/00gbJZ0OdcNs9zi288)<br>
[![Donate via Bitcoin](https://img.shields.io/badge/Donate-Bitcoin-green.svg)](bitcoin:37fsp7qQKU8XoHZGRQvVzQVP8FrEJ73cSJ)<br>
[![Donate via Paypal](https://img.shields.io/badge/Donate-Paypal-green.svg)](https://buy.stripe.com/00gbJZ0OdcNs9zi288)

A Lua version (with ImGui) of the Javsacript version (based on Emscripten+Lua Symmath) of the Lua version (Symmath-based) of the metric visualization and diff geom calculation tool.
Javascript version is here http://christopheremoore.net/metric .

![geodesic along a sphere](images/geodesic.png)

![Gaussian curvature of a torus](images/gaussian.png)

![Ricci curvature on a hyperboloid in one direction](images/ricci%201.png)

![Ricci curvature on a hyperboloid in the other direction](images/ricci%202.png)

TODO
- For Gaussian & Ricci curvatures: compile the equations to OpenCL, update the curvature values to a buffer, reduce to find min and max on that buffer (instead of searching for it via the CPU like I do now)
- Hide geodesics when showing Gaussian, Ricci, Riemann curvature.  Only show arrows in the selected directions for Ricci and Riemann curvatures.  Don't have a 'direction' picker for Gaussian curvature, and add a 2nd one for Riemann curvature. 

