# Harris corner detection
## Paper
[a combined corner and edge detector](https://bmva-archive.org.uk/bmvc/1988/avc-88-023.pdf)
## History
### **1. The Inventors**
* **Who:** Chris Harris and Mike Stephens.
* **When:** 1988.
* **Where:** Plessey Research (United Kingdom).
* **Paper Title:** *"A Combined Corner and Edge Detector."*

### **2. The Problem: Why did they invent it?**
Before 1988, scientists used the **Moravec Corner Detector**. It had a major flaw:
* **The "Stiff Neck" Problem:** Moravec's algorithm could only check for changes in 8 specific directions (horizontal, vertical, and diagonals).
* **The Failure:** If an edge was tilted at an odd angle (e.g., $15^{\circ}$), the detector would miss it. It was not "rotation invariant."

Harris and Stephens needed a way to detect corners accurately **regardless of rotation** for 3D robot vision.

### **3. The Solution: How did they do it?**
They improved Moravec's idea using two mathematical tricks:

#### **Trick A: Calculus instead of Shifting (The Taylor Series)**
Instead of physically moving the window pixel-by-pixel to check for changes, they used **Calculus** (specifically the *Taylor Series Expansion*).
* **Result:** This allowed them to mathematically predict intensity changes in **every possible direction**, not just 8.

#### **Trick B: The "Lazy" Shortcut (The Score Formula)**
In 1988, computers were slow. Calculating the exact "shape" of the data (using *Eigenvalues*) was too computationally expensive.
* **The Hack:** Harris realized he didn't need the exact Eigenvalues. He could estimate the "corner strength" using a simple formula involving multiplication and subtraction.
* **The Formula:**
    $$R = \det(M) - k \cdot (\text{trace}(M))^2$$
    * If $R$ is **Big Positive**: It is a **Corner**.
    * If $R$ is **Big Negative**: It is an **Edge**.
    * If $R$ is **Small**: It is **Flat**.


## Code
01_cornerHarris.py 02_cornerSubPix.py


