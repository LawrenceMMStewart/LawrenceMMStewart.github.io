---
layout: distill
title: Differentiable Clustering with Perturbed Random Forests
description: A simple intro!
tags: differentiable clustering research 
giscus_comments: true
date: 2023-11-29
featured: true

authors:
  - name: Lawrence Stewart 
    url: "https://lawrencemmstewart.github.io/"
    affiliations:
      name: Ecole Normale Superieure, INRIA Paris

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Clustering
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Kruskal's Algorithm
    subsections:
        -name: Recap of Spanning Forests and Trees
        -name: Single Linkagle Clustering
  - name: Differentiable Clustering via Perturbations

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---


## Clustering

Clustering is one of the most classical and commonplace tasks in Machine Learning.  The goal is to separate data $$x_1, \ldots, x_n$$ into $$k$$ groups, which are refered to as clusters. Clustering has wide-spread applications in bio-informatics, data compression, graphics, unsupervised and semi-supervised learning, as well as many other domains! 

There is a large collection of [well established clustering algorithms](https://en.wikipedia.org/wiki/Cluster_analysis) (with a select few being displayed in the table below).

| Methodology        | Examples |  Possible Drawbacks
| ------------- |-------------| -------------| 
| centroid | k-means | NP-Hard, heuristic (not direct solve of objective function).
| connectivity | Linkage Algorithms (e.g. Single, UPGMA)| Computational Costly, Outliers.
| distribution| EM Gaussian| Overfitting, Assumptions.

When dealing with semantic data e.g. Images or Text, applying such algorithms to the data directly is unlikely to lead to meaningful clusters.<d-footnote> To see this, try applying k-means directly on the MNIST data set </d-footnote>! Instead, we would like to learn representations of are data e.g. features of a Neural Network, which when clustered, lead to clusters which capture meaningful semantic information.

Unfortunately, we cannot just plug any classical clustering algorithm into a Deep Learning pipeline<d-footnote> Or more generally, gradient based learning pipeline.</d-footnote> :dizzy_face: .

Can you think of the reason as to why not? The answer is in the box below: 


{% details :warning: Gradient Based Learning Compatibility with Classical Clustering? %}
As a function, the solution of a clustering problem is piece-wise constant with respect to its inputs (such as a similarity or distance matrix), and its **gradient would therefore be zero almost everywhere**. This operation is therefore naturally ill-suited to the use of gradient-based approaches to minimize an objective, such as backpropagation for training Neural Networks.

If the above is not clear at first, just not that the cluster assignment of $$x_i$$ will almost always be the same as the cluster assignment for $$x_i + \epsilon$$ where $$\epsilon$$ denotes an infinitesimal change, so the gradient will be zero.
{% enddetails %}

## Goal :rocket:

In this blog post we will give a simple explanation of our recent work that aims to address the above problem. We will keep math and other technical details to an absolute minimum, but for a more complete picture you can refer to the paper <d-cite key="stewart2023differentiable"></d-cite>.

:pencil2: For any further questions, please feel free to contact me !


## Kruskal's Algorithm

#### Viewing our data as a graph

Firstly, we will recap maximum weight spanning forests and kruskals algorithm.

We can think of our data $$x_1, \ldots, x_n \in \mathbb{R}^d$$ as nodes of a fully-connected graph $$\mathcal{G}$$, where the weight of an edge $$(i,j)$$ is given by the $$(i,j)^{th}$$ entry of a user-chosen similarity matrix $$\Sigma \in \mathbb{R}^{n\times n}$$. A large value of $$\Sigma_{i,j}$$ means that points $$i$$ and $$j$$ are similar, whilst a smaller value means that the points are disimilar.

<!-- There are many possible choices of similarity matrix, for example:

$$
\begin{align}
\Sigma_{ij} &= - \|x_i - x_j\|_2^2 \\[1em]
\Sigma_{ij} &=\exp\left( -\frac{1}{\gamma^2} \|x_i - x_j\|_2^2\right)
\end{align}
$$ -->

Below is an example graph for two different typical choices of $$\Sigma$$.

{% include figure.html path="assets/img/blog-differentiableclustering/Kn.svg" class="img-fluid rounded z-depth-0" zoomable=true %}



<!-- \begin{align}
\Sigma_{ij} &= - \|x_i - x_j\|_2^2 \\
\Sigma_{ij} &=\exp\left( -\frac{1}{\gamma^2} \|x_i - x_j\|_2^2\right)
\end{align} -->

<!-- $$\Sigma_{ij} = - \|x_i - x_j\|_2^2$$

$$\Sigma_{ij} =\exp\left( -\frac{1}{\gamma^2} \|x_i - x_j\|_2^2\right)$$ -->

### Spanning trees

For the complete graph with $$n$$ vertices $$K_n$$ over nodes $$\{x_1, \ldots, x_n\}$$, we denote by $$\mathcal{T}$$ the set of *spanning trees* on $$K_n$$, i.e., subgraphs with no cycles and one connected component. Among these trees will be one or more that has maximum weight (the total weight of all edges in the tree), which is known as the *maximum weight spanning tree*.

`Kruskals algorithm` is a greedy algorithm to find a maximum weight spanning tree. It is incredibly simple, and consists of adding edges in a greedy manner to build the tree, and ignoring an edge if it would lead to a cycle. The psuedo-code for the algorithm is as follows:

{% highlight python %}
# 
tree = {} 
edges = sort(edges)
for e in edges:
  if union(tree, {e}) has no cycle:
    tree = union(tree, {e})
  else:
    pass
  if tree is spanning:
    break
{% endhighlight %}

At each time step $t$, we will have a forest with $k=n-t$ connected components, where $n$ is the number of data points / nodes in the graph. A visual depiction of the algorithm in action can be seen below:

{% include figure.html path="assets/img/blog-differentiableclustering/mst.gif" class="img-fluid rounded z-depth-0" zoomable=true loop=true style="width:90%;" %}

### Relationship to Clustering

When running Kruskal's algorithm, one typically builds the tree $$T$$ by keeping track of the adjacency matrix $$A\in \{0, 1\}^{n\times n}$$ of the forest at each time step. We recall that:

$$
\begin{equation*}
  A_{i,j} = 1 \Longleftrightarrow (i,j)\in T
\end{equation*}
$$

However by modifying Kruskal's algorithm, we can also keep track of the cluster equivalence matrix $$M\in \{0, 1\}^{n\times n}$$, where:

$$
\begin{equation*}
  M_{i,j} = 1 \Longleftrightarrow (i, j) \text{ are both in the same connected component of }  T
\end{equation*}
$$

If we are to stop Kruskal's algorithm one step before completion, we will obtain a forest with $$k=2$$ connected components. We can view these two connected components as clusters!

More generally, if we are stop the algorithm $$k+1$$ steps before completion, we will obtain a forest with $$k$$ connected components. Whats nice, is it turns out that Kruskal's algorithm has a **Matroid Structure**, when means that if we stop the algorithm when the forest has $$k$$ connected components, that forest will indeed have maximum weight amongst all forests of $$\mathcal{G}$$ that have $$k$$ connected components!  More details are given in the box below, but they are not neccessary to understand the goal of this blog.

{% details Optimality of Kruskal's %}
The greedy optimality of Kruskal's follows from the fact that the forests of $$\mathcal{G}$$ correspond to independent sets of the [Graphic Matroid](https://en.wikipedia.org/wiki/Graphic_matroid). 

To verify this is true, note that the intersection of two forests is always a forest, and the spanning trees of a graph form the basis for the matroid. The matroid circuits can are the cycles in the graph. Optimality of Kruskal's follows trivially (as the algorithm is equivalent to finding the maximum weight basis of the graph matroid).
{% enddetails %}

Hence we can use Kruskal's algorithm to cluster our data into $$k$$ groups:

{% highlight python %}
def cluster(Sigma, k)
  do:
    run kruskals step until k connected components
  return: A_k, M_k
{% endhighlight %}

This algorithm is known as **Single-Linkage** and is related to a family of *hierarchical clustering* algorithms. An example of the algorithm running is given below, where the data is separated into three distinct clusters:

{% include figure.html path="assets/img/blog-differentiableclustering/kruskals.gif" class="img-fluid rounded z-depth-0" zoomable=true loop=true style="width:90%;" %}

This is all well and good, but what does the above have anything to do with differentiable clustering?

### Perturbations

We will now take a pause from Kruskal's algorithm to look at **perturbations**, sometimes also called **randomized smoothing**. If maths isn't your thing, not to worry as understanding the perturbations / smoothing in detail is not neccessary for getting a grasp of the overall methodology.

For a [convex hull](https://en.wikipedia.org/wiki/Convex_hull) $$\mathcal{C} \subset \mathbb{R}^d$$ we define:

- **argmax solution** $$y^*:\mathbb{R}^d \rightarrow \mathbb{R}^d$$ 
- **max solution** $$F:\mathbb{R}^d\rightarrow\mathbb{R}$$ 

as follows:

$$
\DeclareMathOperator{\argmax}{argmax}
\begin{align}
 y^*(\theta) &= \argmax\limits_{y \in \mathcal{C}} \langle y, \theta \rangle. \\[1em]
 F(\theta) &= \max\limits_{y \in \mathcal{C}} \langle y, \theta \rangle. 
\end{align}
$$

Note both $$y^*$$ and $$F$$ are piece-wise constant in $$\theta$$. For this reason both the gradient $$\nabla_\theta F(\theta) \in \mathbb{R}^d$$ and Jacobian $$J_\theta y^*(\theta)\in \mathbb{R}^{d\times d}$$ will be zero almost everywhere, similar to the case of classical clustering algorithms.

To get non-zero gradients, we can instead replace $$\theta$$ in the above, by $$\theta + \epsilon Z$$, where $$Z$$ is some an exponential-family random variable e.g. multi-variate Gaussian with zero mean and identity covariance. This induces a probability distribution:

$$\mathbb{P}_Y(Y = y ; \theta) = \mathbb{P}_Z(y^*(\theta + \epsilon Z) = y)$$

This yields their perturbed versions:

$$
\DeclareMathOperator{\argmax}{argmax}
\begin{align}
 y^*_\epsilon(\theta) &= \textcolor{orange}{\mathbb{E}_Z}\left[\argmax\limits_{y \in \mathcal{C}} \langle y, \theta + \epsilon \textcolor{orange}{Z} \rangle\right]. \\[1em]
 F_\epsilon(\theta) &= \textcolor{orange}{\mathbb{E}_Z}\left[\max\limits_{y \in \mathcal{C}} \langle y, \theta + \epsilon \textcolor{orange}{Z} \rangle \right].
\end{align}
$$

### Recap of Spanning Forests and Trees
### Single Linkage Clustering

## Differentiable Clustering via Perturbations


## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$


Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) that brought a significant improvement to the loading and rendering speed, which is now [on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).

***

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

## Code Blocks

Syntax highlighting is provided within `<d-code>` tags.
An example of inline code snippets: `<d-code language="html">let x = 10;</d-code>`.
For larger blocks of code, add a `block` attribute:

<d-code block language="javascript">
  var x = 25;
  function(x) {
    return x * x;
  }
</d-code>

**Note:** `<d-code>` blocks do not look good in the dark mode.
You can always use the default code-highlight using the `highlight` liquid tag:

{% highlight javascript %}
var x = 25;
function(x) {
  return x * x;
}
{% endhighlight %}

***

## Interactive Plots

You can add interative plots using plotly + iframes :framed_picture:

<div class="l-page">
  <iframe src="{{ '/assets/plotly/demo.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

The plot must be generated separately and saved into an HTML file.
To generate the plot that you see above, you can use the following code snippet:

{% highlight python %}
import pandas as pd
import plotly.express as px
df = pd.read_csv(
  'https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv'
)
fig = px.density_mapbox(
  df,
  lat='Latitude',
  lon='Longitude',
  z='Magnitude',
  radius=10,
  center=dict(lat=0, lon=180),
  zoom=0,
  mapbox_style="stamen-terrain",
)
fig.show()
fig.write_html('assets/plotly/demo.html')
{% endhighlight %}

***

## Details boxes

Details boxes are collapsible boxes which hide additional information from the user. They can be added with the `details` liquid tag:

{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

***

## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list.
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

* Unordered list can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
http://www.example.com or <http://www.example.com> and sometimes
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style:
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

```python
s = "Python syntax highlighting"
print s
```

```
No language indicated, so no syntax highlighting.
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote.


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
