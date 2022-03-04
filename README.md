# Neural Style Transfer

![Comparison](https://github.com/julianbel/neural-style-transfer/blob/master/comparison.png?raw=true)

Implementación en Tensorflow de la técnica Neural Style Transfer, propuesta en el paper titulado [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al.).

Se trata de una técnica de optimización que consiste en tomar dos imágenes, una que proporciona el contenido y otra la referencia de estilo (como una obra artística de un pintor), para fusionarlas en un único output que luce como la imagen de contenido, "pintada" en el estilo de la imagen que proporcionó dicha referencia.

Esta técnica se implementa a través de la optimización de la imagen de output hasta alcanzar la representación estadística del contenido y estilo de las respectivas imágenes. Estas representaciones se extraen de las imágenes originales a partir de las capas intermedias de una Red Neuronal Convolucional (CNN). Iniciando desde la capa de input, los primeros mapas de activación representan características de bajo nivel, como bordes y texturas, y a medida que se va avanzando a lo largo de la red, las capas finales contienen representaciones de características de alto nivel, como ojos en un rostro.

## Representación del Contenido

Dado que se utiliza una CNN pre-entrenada, se puede capturar la representación del contenido de una imagen directamente tomando los mapas de activación de las capas intermedias capaces de capturar las características más complejas (de alto nivel) de las mismas.

## Representación del Estilo

Para capturar el estilo artístico de una imagen se utiliza la matriz de Gram.

La matriz de Gram es una representación del estilo de una imagen que computa las correlaciones entre las diferentes respuestas de los filtros de las capas convolucionales. Se obtiene realizando el producto punto de cada mapa de activación consigo mismo en cada ubicación, promediando ese producto en todas las ubicaciones, de acuerdo a la siguiente fórmula:

<img src="https://render.githubusercontent.com/render/math?math={\Large G^{l}_{cd}=\frac{\sum_{ik}F^{l}_{ijc}(x)F^{l}_{ijc}(x)}{IJ}}#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\Large \color{white}G^{l}_{cd}=\frac{\sum_{ik}F^{l}_{ijc}(x)F^{l}_{ijc}(x)}{IJ}}#gh-dark-mode-only">

## Funciones de Costo

La técnica de Neural Style Transfer se vale de dos funciones de costo y un término de regularización que se minimizan en conjunto a través del proceso de backpropagation para capturar el contenido de la imagen original (a través de la _Content Loss_) y el estilo artístico de otra (a través de la _Style Loss_) para lograr reflejarlo en el output de forma combinada.

Para generar una imagen que combine el contenido de una fotografía con el estilo artístico de una pintura resulta necesario minimizar conjuntamente tanto la distancia de una imagen de ruido blanco con respecto a una representación de contenido de una fotografía (Content Loss) en una de las capas de la CNN y la representación del estilo de la pintura (Style Loss) capturada en las restantes capas convolucionales, ajustadas por un término de regularización (Total Variation). En términos formales, la función de costo total a minimizar es:

<img src="https://render.githubusercontent.com/render/math?math={\Large \mathcal{L}_{total}(\overrightarrow{p},\overrightarrow{a},\overrightarrow{x}) = \alpha \mathcal{L}_{content}(\overrightarrow{p},\overrightarrow{x})+\beta \mathcal{L}_{style}(\overrightarrow{a},\overrightarrow{x})+\lambda_{TV}\mathcal{L}_{TV}(\overrightarrow{x})}#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\Large  \color{white}\mathcal{L}_{total}(\overrightarrow{p},\overrightarrow{a},\overrightarrow{x}) = \alpha \mathcal{L}_{content}(\overrightarrow{p},\overrightarrow{x})+\beta \mathcal{L}_{style}(\overrightarrow{a},\overrightarrow{x})+\lambda_{TV}\mathcal{L}_{TV}(\overrightarrow{x})}#gh-dark-mode-only">

Donde:
- <img src="https://render.githubusercontent.com/render/math?math={\large \alpha}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math={\large \color{white}\alpha}#gh-dark-mode-only">: coeficiente de énfasis de la reconstrucción del contenido.
- <img src="https://render.githubusercontent.com/render/math?math={\large \beta}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math={\large \color{white}\beta}#gh-dark-mode-only">: coeficiente de énfasis de la reconstrucción del estilo.
- <img src="https://render.githubusercontent.com/render/math?math={\large \lambda_{TV}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math={\large \color{white}\lambda_{TV}}#gh-dark-mode-only">: ratio que expresa el énfasis entre contenido y estilo (alfa / beta).

En esencia, a medida que una imagen va pasando por las sucesivas capas de una CNN, se van almacenando las respuestas en una matriz _N x M_, donde N simboliza el número de filtros en una capa y M el tamaño de cada filtro (alto x ancho). 

Durante el proceso de backpropagation, las losses mencionadas se van calculando y minimizando de la siguiente manera:

### Style Loss

En líneas generales, al alimentar la CNN con una imagen que contenga el estilo artístico deseado y una imagen de entrada, esta devolverá los outputs de las capas intermedias del modelo. Sin embargo, para el cómputo de la Style Loss, la técnica de Neural Style Transfer se vale de las matrices de Gram. En lugar de comparar dichos outputs tal cual resultan de la aplicación de los filtros convolucionales, se comparan sus matrices de Gram.

En términos matemáticos, se puede describir la Style Loss de una imagen de entrada y la de estilo como la distancia cuadrática media entre la representación del estilo (capturada en la matriz de Gram) de cada una de ellas. La representación del estilo de una imagen puede describirse como la correlación entre las diferentes respuestas de los filtros capturada en la matriz de Gram.

Para replicar el estilo artístico de la imagen de estilo en la imagen de entrada, es necesario realizar Gradient Descent con el objetivo de transformar la imagen de entrada en una cuya representación de estilo coincida con la imagen de estilo original.

### Content Loss

La Content Loss sigue un principio similar a la Style Loss sólo que, en lugar de emplear matrices de Gram, simplemente se calcula la distancia euclideana entre las dos representaciones de una imágen de entrada y otra con el contenido que se desea replicar.

Resulta importante destacar que a diferencia de una tarea de clasificación utilizando una CNN, el cómputo del gradiente se realiza con respecto a los píxeles de la imagen de entrada en lugar de los pesos, que se mantienen constantes a lo largo del proceso de backpropagation. Esto permite la utilización de arquitecturas preentrenadas a través de Transfer Learning, dado que sólo se emplearán los filtros de las capas convolucionales para generar la imagen combinada. En el caso particular del paper bajo análisis, se utiliza la VGG19 pasando el argumento _include_top = False_, de manera de excluir las capas densas de la instanciación del modelo.

### Total Variation Loss

Adicionalmente, se aplica al cálculo de la loss un término de regularización que apunta a reducir los artefactos de alta frecuencia (i.e. ruido, como por ejemplo píxeles muy claros o muy oscuros) que puedan manifestarse en la imagen como consecuencia del proceso de transferencia de estilo.

## Optimización

El paper propone la minimización de la función de costo a partir de la versión básica del algoritmo L-BFGS (_Low-memory Broyden–Fletcher–Goldfarb–Shanno_), el cual constituye un método de optimización quasi-Newton de funciones con una gran cantidad de parámetros haciendo un uso limitado de memoria, que se logra usando un estimado de la inversa de la matriz Hessiana para guiar el proceso de optimización. Mientras que el algoritmo BFGS almacena una aproximación densa de _n x n_ de esta matriz (donde _n_ representa el número de variables del problema), L-BFGS almacena una cantidad limitada de vectores que representan dicha aproximación de forma implícita.

Alternativamente, se pueden utilizar algoritmos adaptativos de Gradient Descent como _Adam_ o _AdaGrad_ para llevar adelante el mismo proceso de minimización conjunta de las funciones de costo. La diferencia principal entre estos algoritmos y L-BFGS radica en que son computacionalmente más rápidos en su ejecución. L-BFGS es un método que estima la curvatura del espacio de parámetros, lo que lo hace más robusto que algoritmos adaptativos cuando el espacio de parámetros presenta muchos "_saddle-points_" pero los vuelve más lentos y costosos computacionalmente.
