# Rede Neural Multi-camada
## Modelagem da porta lógica XOR usando back-propagation

- [Introdução](#intro)
    - [Back-propagation](#backprop)
- [Método](#method)
    - [Função Custo](#costfunc)
    - [Função de Ativação](#activfunc)
    - [Atualização de Pesos e Vieses](#weights)
    - [Inicialização de Pesos e Vieses](#init)
- [Guia de Uso](#guide)
- [Resultado](#result)
- [Comparação com Perceptron de Camada Única](#compare)
- [Discussão](#discussion)
- [Referências](#ref)


# Introdução <a name="intro"></a>
O nome redes neurais no âmbito da computação toma emprestado o termo de inspiração da neurociência não necessariamente por conta de neurônios reais mas sim a sua aplicação em redes artificiais. Para evitar confusões com termos da área biológica, entretanto, outros termos como sinapses são evitados, ainda que se associe nodes, ou células onde as operações ocorrem, a neurônios. Isso deve, principalmente, ao início dos estudos em computação dessas estruturas que tinha como foco a modelagem de neurônios cerebrais de fato.  

Partindo de modelos bem rudimentares e inspirados nos próprios neurônios, pôde-se obter formas bem eficientes de resolver alguns problemas como é o caso dos problemas aqui a serem apresentados, modelagem de portas lógicas. 

O tipo de "neurônio" mais simples é o de classificação binária, chamado de perceptron após a sua implementação por Frank Rosenblatt em 1957. Proposto por Warren McCulloch e Walter Pitts em 1943, consistia em computar uma soma dos pesos do seu input de outras unidades e retorna um ou zero caso a soma dos pesos estiver acima ou abaixo de um determinado valor (limiar), de forma prática:

$$
n_{i}(t+1)=\Theta\left(\sum_{i} w_{i j} n_{j}(t)-\mu_{i}\right).
$$

Onde $n_{i} = 1$ ou $n_{i} = 0$, indicando o estado neurônio disparar ou não, $\Theta$ a função de Heaviside e $\mu_{i}$ é o limiar. O peso $w_{i j}$ representa a intensidade de uma sinapse do neurônio $j$ para o neurônio $i$, podendo ser positivo para excitações ou negativo para inibições. 

<p align="center">
    <img"https://i.imgur.com/tos95fn.png"/>
</p>
    

McCulloch e Pitts provaram então que o esse modelo era capaz de realizar qualquer operação computacional, dados os pesos adequados. 

Ressalta-se que um neurônio real apresenta diferenças significantes do perceptron. Em suma, os inputs} não são discretos os outputs não são unitários mas sim sequências de pulsos. 

Ainda assim, a quebra de simetria ou ausência de linearidade entre os $inputs$ e $outputs$ tanto do perceptron quanto do neurônio real aparenta ser o fator em comum que confere a eles grande eficiência.  

Para os modelos mais atuais de redes neurais, $n_{i}$ passa a ser um valor contínuo e função de Heaviside $\Theta$ é trocada por uma função mais geral e é também contínua chamada de função de ativação.

Um problema que perceptrons de camada única não são capazes de resolver é o de classificação binária do tipo XOR e equivalentes. 


Contudo, perceptrons de camadas múltiplas onde há "comunicação" entre as camadas anteriores e seguintes e vice-versa são capazes de resolver e obter resultados que reproduzem a porta lógica XOR. Esse modelo é conhecido como back-propagation. 



## Back-Propagation <a name="backprop"></a>


Em perceptrons multi-camadas, as camadas existentes que intermediam as camadas de input e output são chamadas de camadas ocultas. Para o caso da porta XOR, apenas uma camada oculta se mostra suficiente para classificação binária esperada. 

Para que o resultado pudesse ser obtido foi desenvolvida uma regra de aprendizagem chamada de back-propagation, que consiste em "informar" ou propagar as correções dos pesos para as camadas anteriores após a propagação direta (para as camadas seguintes). 

Dessa forma, o algoritmo implementa um aprendizado de pares de input-ouput e usa como base o método do gradiente descendente. 

Partindo da minimização de uma função custo $\mathcal{C}(W,b)$, que determina a taxa com que o erro se propaga na rede.

Assim, os pesos $W$ e vieses $b$ são atualizados de acordo com:

$$
\Delta W_{k k^{\prime}}^{\ell} =\frac{-\eta}{N_{s}} \sum_{s=1}^{N_{s}} \frac{\partial C_{s}}{\partial W_{k k^{\prime}}^{\ell}},
$$

$$
\Delta b_{k}^{\ell} =\frac{-\eta}{N_{s}} \sum_{s=1}^{N_{s}} \frac{\partial C_{s}}{\partial b_{k}^{\ell}}.
$$

Em que $N_{s}$ é o número de elementos da sequência de input(s)}, $s$ o elemento e $C_{s}$ a função custo associada ao elemento $s$. Os índices de $W$ e $b$ são $k$ o neurônio da camada $l$ e $k\prime$ o neurônio da camada $l-1$ de origem. 



# Método <a name="method"></a>

Para a modelagem da porta XOR:

| $s$ | $x_{1}$ | $x_{2}$ | $y$ |
|---|----|----|---|
| 1 | 0  | 0  | 0 |
| 2 | 1  | 0  | 1 |
| 3 | 0  | 1  | 1 |
| 4 | 1  | 1  | 0 |

foram determinados duas sequências de inputs para o aprendizado, $X_{1} = \{x_{1}(1), x_{1}(2), x_{1}(3), x_{1}(4)\}$ e $X_{2} = \{x_{2}(1), x_{2}(2), x_{2}(3), x_{2}(4)\}$, e uma única de $Y = \{y(1), y(2), y(3), y(4)\}$ conforme a tabela verdade acima.

## Função Custo <a name="costfunc"></a>

A função custo total utilizada foi a log-loss somada sobre todos os elementos das sequências de inputs.

$$
\mathcal{C}(\mathbf{W}, \mathbf{b})=\frac{1}{N_{s}} \sum_{s=1}^{N_{s}}\left[-y(s) \ln \left(Y_{\text {out }}(s)\right)-(1-y(s)) \ln \left(1-Y_{\text {out }}(s)\right)\right],
$$

o que confere uma boa implementação de um critério de convergência já que a função log-loss fornece a proximidade entre os dados produzidos $Y_{out}$ e os esperados $y$.

## Função de Ativação <a name="activfunc"></a>

A função de ativação para este problema foi determinada como a função sigmoide. 

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

<p align="center">
    <img src="https://i.imgur.com/9qCTi4D.png"/>
</p>

## Atualização dos Pesos e Vieses <a name="weights"></a>

Definindo as quantidades:

$$
\delta_{k}^{\ell} = \frac{\partial C_{s}}{\partial Z_{k}^{\ell}}, \qquad Z_{k}^{\ell} = \left(\sum_{k^{\prime} \in \ell-1} W_{k k^{\prime}}^{\ell} Y_{k^{\prime}}^{\ell-1}\right)+b_{k}^{\ell}
$$

e usando a regra da cadeia

$$
\delta_{k}^{\ell} = \frac{\partial C_{s}}{\partial Z_{k}^{\ell}} = \frac{\partial C_{s}}{\partial Y_{k}^{\ell}} \cdot  \frac{\partial Y_{k}^{\ell}}{\partial Z_{k}^{\ell}} =\frac{\partial C_{s}}{\partial Y_{k}^{\ell}} \sigma^{\prime}\left(Z_{k}^{\ell}\right), \qquad \sigma (Z_{k}^{\ell}) = Y_{k}^{\ell}
$$

$$
\begin{aligned}
\delta_{1}^{L}&=\frac{\partial C_{s}}{\partial Y_{1}^{L}} \sigma^{\prime}(Z_{1}^{L})=\frac{\partial C_{s}}{\partial Y_{\text {out }}}\left(1-Y_{\text {out }}\right) Y_{\text {out }}\\
&=\frac{\partial}{\partial Y_{\text {out }}}\left[-y(s) \ln \left(Y_{\text {out }}(s)\right)-(1-y(s)) \ln \left(1-Y_{\text {out }}(s)\right)\right] \left(1-Y_{\text {out }}\right) Y_{\text {out }}\\
&=\left[-y \frac{1}{Y_{\text {out }}}-(1-y) \frac{(-1)}{\left(1-Y_{\text {out }}\right)}\right]\left(1-Y_{\text {out }}\right) Y_{\text {out }}\\
&=-y\left(1-Y_{\text {out }}\right)+(1-y) Y_{\text {out }}\\
&=-y+Y_{\text {out }} y+Y_{\text {out }}-y Y_{\text {out }}\\
&=Y_{\text {out }}-y
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial Z_{1}^{L}}{\partial Y_{k}^{2}}&=\sum_{k^{\prime}} W_{1 k^{\prime}}^{L}\\
\frac{\partial \ln (\sigma\left(Z^{\ell} k\right))}{\partial Y_{k}^{2}}&=\frac{1}{\sigma\left(Z_{k}^{k}\right)} \sigma^{\prime}\left(Z_{k}^{\ell}\right) \frac{\partial Z_{k}^{\ell}}{\partial Y_{k}^{\ell}}=\sum_{k^{\prime}} W_{1 k^{\prime}}^{L} \frac{\sigma^{\prime}\left(Z_{k}^{\ell}\right)}{\sigma\left(Z_{k}^{\ell}\right)}=\sum_{k^{\prime}} W_{1 k^\prime}^{L}\left(1-\sigma\left(Z_{k}^{\ell}\right)\right)\\
\frac{\partial \ln\left(1-\sigma\left(Z_{k}^{\ell}\right)\right)}{\partial Y_{k}^{\ell}}&=\frac{1}{\left(1-\sigma\left(Z_{k}^{\ell}\right)\right)}\left(-\sigma^{\prime}\left(Z_{k}^{\ell}\right)\right) \frac{\partial Z_{k}^{\ell}}{\partial Y_{k}^{\ell}}=\sum_{k^{\prime}} W_{1 k^\prime}^{L}{ }^{L}\frac{(-\sigma^{\prime}\left(Z_{k}^{\ell}\right))}{\left(1-\sigma\left(Z_{k}^{\ell}\right)\right)}=\sum_{k^{\prime}} W_{k^{\prime}}^{L} \sigma\left(Z_{k}^{\ell}\right),
\end{aligned}
$$


$$
\begin{aligned}
\frac{\partial C_{s}}{\partial Y_{k}^{\ell}} &= \sigma^{\prime}\left(Z_{k}^{\ell}\right), \qquad Z_{k}^{\ell}=\sum_{k^{\prime}} W_{1 k}^{L} Y_{k}^{\ell}+b_{k} \\
\end{aligned}
$$

$$
\begin{aligned}
\delta_k^{\ell} & =\sigma^{\prime}\left(Z_k^{\ell}\right) \frac{\partial}{\partial Y_k^{\ell}}\left[-y \ln \left(\sigma\left(\sum_{k^{\prime}} W_{1 k^{\prime}}^L Y_k^{\ell}+b_k\right)\right)-(1-y) \ln \left(1-\sigma\left(\sum_{k^{\prime}} W_{1 k^{\prime}}^L Y_k^{\ell}+b_k\right)\right)\right] \\
& =\sigma^{\prime}\left(Z_k^{\ell}\right)\left[-y \sum_{k^{\prime}}^L W_{1 k^{\prime}}^L\left(1-Y_k^{\ell}\right)+\sum_{k^{\prime}} W_{1 k^{\prime}}^L(1-y)\left(Y_k^{\ell}\right)\right] \\
& =\sigma^{\prime}\left(Z_k^{\ell}\right)\left[\sum_{k^{\prime}} W_{1 k^{\prime}}^L\left(Y_k^{\ell}-y\right)\right] \\
& =\sigma^{\prime}\left(Z_k^{\ell}\right)\left[W_{1 k}^L\left(Y_{\mathrm{out}}-y\right)\right] \\
& =\sigma^{\prime}\left(Z_k^{\ell}\right)\left[W_{1 k}^L \delta_1^L\right]
\end{aligned}
$$


$$
\frac{\partial Z_{k}^{\ell}}{\partial b_{k}^{\ell}}=\frac{\partial}{\partial b_{k}^{\ell}}\left(\sum_{k^{\prime}} W_{k k^{\prime}}^{\ell} Y_{k}^{\ell-1}+b_{k}^{\ell}\right)=1,  \qquad  \frac{\partial Z_{k}^{\ell}}{\partial W_{k k^{\prime}}^{\ell}}= \frac{\partial}{\partial W_{k k^{\prime}}^{\ell}}\left(\sum_{k^{\prime}} W_{k k^{\prime}}^{\ell} Y_{k^{\prime}}^{\ell-1}+b_{k}^{\ell}\right)=Y_{k^{\prime}}^{\ell-1}, \\
$$

$$
\begin{aligned}
\frac{\partial C_{s}}{\partial b_{k}^{\ell}}&=\frac{\partial C_{s}}{\partial Z_{k}^{\ell}} \frac{\partial Z_{k^{k}}^{\ell}}{\partial b_{k}^{\ell}} = \frac{C_{s}}{\partial Y_{k}^{\ell}} \sigma^{\prime}\left(Z_{k}^{\ell}\right) =\delta_{k}^{\ell} \\
\frac{\partial C_{s}}{\partial W_{k k^{\prime}}^{\ell}}&=\frac{\partial C_{s}}{\partial Z_{k}^{\ell}} \frac{\partial Z_{k}^{\ell}}{\partial W_{k k^{\prime}}^{\ell}}=\frac{\partial C_{s}}{\partial Y_{k}^{\ell}} \sigma^{\prime}\left(Z_{k}^{\ell}\right) Y_{k^{\prime}}^{\ell-1}=\delta_{k}^{\ell} Y_{k^{\prime}}^{\ell-1}.
\end{aligned}
$$

Logo a atualização dos pesos é dada por:


$$
\begin{aligned}
W_{k k^{\prime}}^{\ell} & \rightarrow W_{k k^{\prime}}^{\ell}-\frac{\eta}{N_s} \sum_s^{N_s} \delta_k^{\ell} Y_k^{\ell-1} \\
b_k^{\ell} & \rightarrow b_k^{\ell}-\frac{\eta}{N_s} \sum_s^{N_s} \delta_k^{\ell} .
\end{aligned}
$$

Onde $\eta$ é a taxa de aprendizado (learning rate) da rede. 

## Inicialização de Pesos e Vieses <a name="init"></a>

Um problema amplamente estudado é o de escolha dos pesos iniciais que por não poderem ser 0 pois não quebram a simetria caso sejam. 

Logo, é necessário algum processo de otimização da escolha desses pesos. A racionalização mais comum é de que os pesos devem ser suficientemente grandes para que a informação seja propagada corretamente porém pesos muito grandes podem provocar irregularidades como overflows. Além disso, algoritmos como este, que faz uso do gradiente descendente, tendem a parar em regiões próximas dos parâmetros iniciais, implicando o uso de pesos que não devem alterar muito os parâmetros iniciais em relação aos resultados esperados.

Uma abordagem que possui resultados expressivos em redes neurais que fazem uso da função de ativação do tipo sigmoide é a inicialização de pesos de Xavier:

$$
W \sim U\left(-\frac{1}{\sqrt{n_{l-1}}}, \frac{1}{\sqrt{n_{l-1}}}\right)
$$


Para este modelo, foi utilizado por padrão a distribuição de pesos de Xavier normalizada, que consiste em um distribuição uniforme do tipo:

$$
W \sim U\left(-\sqrt{\frac{6}{{n_{l-1}+n_{l+1}}}}, \sqrt{\frac{6}{{n_{l-1}+n_{l+1}}}}\right)
$$

Onde $n_{l}$ é o número de neurônios da camada anterior e $n_{l+1}$ o número de neurônios da camada seguinte. 

Em relação aos vieses, a inicialização como zero é razoável nesse caso pois não há contribuição dos vieses no camada de saída que sobrepõe consideravelmente o valor dos pesos, responsáveis pela quebra de simetria.  


# Guia de Uso <a name="guide"></a>

```
> python3 xor_modeling.py --help
usage: xor_modeling.py [-h] [-g porta lógica] [-n neurônios] [--lr aprendizado]
                       [--maxit máximo de iterações] [--conv convergência]
                       [--lim convergência convergência]

Rede Neural de modelagem de porta lógica, utilizando o método do gradiente descendente e back-
propagation com função custo log-loss, função de ativação sigmoide, 1 camada oculta de "n"
neurônios, hiperparâmetro "h", máximo de iterações "maxit" e convergência usando a diferença
sobre a função custo "conv" onde "n", "h", "maxit" e "conv" são parâmetros a serem definido
pelo usuário, tendo valores definidos por padrão. Outro fator que pode ser definido pelo
usuário é o limite inferior e superior da distribuição uniforme dos pesos e vieses (Ws e bs).
Ainda que direcionado à porta XOR, garante a modelagem de outras portas lógicas conhecidas
AND, NOR, OR, NAND, XNOR. Dependências: matplotlib, numpy, argparse*

options:
  -h, --help            show this help message and exit
  -g porta lógica, --gate porta lógica
                        insira a porta lógica desejada. Opções: AND, OR, NOR, NAND, XOR, XNOR.
                        Default = XOR - Argumento do tipo: str
  -n neurônios, --nnum neurônios
                        insira o número de neurônios presentes na camada oculta. Default = 4 -
                        Argumento do tipo: int
  --lr aprendizado      insira a taxa de aprendizado (learning rate). Default = 1.0 -
                        Argumento do tipo: float
  --maxit máximo de iterações
                        insira o número máximo de iterações. Default = 5000 - Argumento do
                        tipo: int
  --conv convergência   insira o valor de convergência para a diferença entre a função custo
                        em relaçao à função custo anterior, basicamente |C(n+1) - C(n)| <
                        "conv". Default = 1e-08 - Argumento do tipo: float
  --lim convergência convergência
                        insira o valor limite inferior e superior da distribuição uniforme dos
                        pesos e vieses (Ws e bs). Default = 0 0. Por padrão, esses valores
                        resultam na Inicialização de Pesos de Xavier Normalizada. Caso
                        especial 1 1: Pesos e vieses sugeridos - Argumento do tipo: float
                        float
```

# Resultado <a name="result"></a>

Exemplo:

```
> python3 xor_modeling.py
Pesos (Weights) e Vieses (Biases) Iniciais W ~ U(-sqrt(6/(n+m)),sqrt(6/(n+m))):

W(l=1)
[[-0.78416706  0.70013829]
 [ 1.274711    0.09583239]
 [ 0.34750134 -0.70974577]
 [-0.10455123 -1.01884815]]

b(l=1)
[0. 0. 0. 0.]

W(l=2)
[0.22142459 1.06778323 0.9526455  0.5042465 ]

b(l=2)
0


Taxa de Aprendizado: 1.0
Neurônios da Camada Oculta: 4
Número de Iterações: 6000 de 6000
Critério de Convergência: |C(n+1) - C(n)| < 1e-08

Tabela de Classificação:

      s  |  X1  |  X2  |  y  |     Y
     -----------------------------------
      1  |   0  |   0  |  0  |  0.00044
      2  |   0  |   1  |  1  |  0.99849
      3  |   1  |   0  |  1  |  0.99897
      4  |   1  |   1  |  0  |  0.0023
```

<p align="center">
    <img src="https://i.imgur.com/mPOJmkB.png"/>
</p>


# Comparação com Perceptron de Camada Única <a name="compare"></a>

Realizando a tentativa de modelagem da porta XOR com um perceptron de camada única, que é equivalente à operações exclusivamente de forward propagation, são obtidos resultados do tipo:

<p align="center">
    <img src="https://i.imgur.com/eKR846S.png"/>
</p>


Resultados consideravelmente homogêneos para todas os inputs. Enquanto que para as portas AND e OR, são obtidos resultados que satisfazem as condições, sendo essas bem classificadas. 

<p align="center">
    <img src="https://i.imgur.com/EI4XmHS.png"/>
</p>


<p align="center">
    <img src="https://i.imgur.com/uY9iaFa.png"/>
</p>


# Discussão <a name="discussion"></a>

É interessante observar que há dois cenários comuns ao utilizar a inicialização de pesos que assume valores negativos, pois ela permite o cenário que permite que a sensibilidade do modelo tenha também uma tendência a classificar a combinação de inputs $(X1 = 0.5, X2 = 0.5)$ como 0, o que não ocorre com o caso dos pesos sugeridos, onde o input $(X1 = 0.5, X2 = 0.5)$ é classificado como 1, o que implica que esse comportamento está diretamente relacionado à escolha dos pesos.

Não há razão para acreditar que essa variação na região $(X1 = 0.5, X2 = 0.5)$ indica alguma falha no modelo, pois o modelo diz respeito somente aos extremos, e com base nas tabelas de classificação e gráficos, a porta XOR é bem representada em ambos os casos.

No que diz respeito à variação do hiperparâmetro que determina a taxa de aprendizado, aqui temos um exemplo no qual atribuir valor maior do que 1 pode ser benéfico visto que as condições propostas garantem a efetividade do gradiente descendente (localizar o mínimo) e, portanto, não oferece risco quanto a atribuir um peso maior as derivadas do que aos próprios pesos. Porém há um valor limite para que ele aumente a precisão da classificação em que o limite é próximo de $\eta = 9$, onde, para valores acima desse limite, a classificação apresenta um underflow, onde valores muito pequenos são tratados como 0, prejudicando as atualizações dos pesos.

A regra de aprendizado empregada no programa apresentado, back-propagation, é necessária por conta da ausência de tendência natural dos outputs que caracterizam a porta XOR, não há preferência instantânea, não sendo suficiente para que um perceptron de camada única seja capaz de modelá-la. Com base no gráfico, existe uma  descontinuidade entre os valores, com duas regiões de mínimos e tal fator implica que o gradiente descendente em uma única camada não é capaz de predições de boa precisão com regiões de múltiplos mínimos ou máximos.


# Referências <a name="ref"></a>

- GIORDANO, N. J., DE JONG, M. L., MCKAY, S. R., AND CHRISTIAN, W. _Computational physics_. Computers
in Physics 11, 4 (1997), 351–351.

- GLOROT, X., AND BENGIO, Y. _Understanding the diﬀiculty of training deep feedforward neural networks.
In Proceedings of the thirteenth international conference on artificial intelligence and statistics_ (2010), JMLR
Workshop and Conference Proceedings, pp. 249–256.

- GOODFELLOW, I., BENGIO, Y., AND COURVILLE, A. _Deep learning (adaptive computation and machine
learning series)_. Cambridge Massachusetts (2017), 321–359.

- HERTZ, J., KROGH, A., AND PALMER, R. G. _Introduction to the theory of neural computation_. CRC Press,
2018
