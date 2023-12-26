# coding = utf-8

import argparse
import numpy as np
import matplotlib.pyplot as plt

"""
    IMPLEMENTAÇÃO DE BACK-PROPAGATION PARA MODELAGEM DE PORTA XOR
"""

parser = argparse.ArgumentParser(
    description='Rede Neural de modelagem de porta lógica, utilizando o método do gradiente descendente e back-propagation com função custo log-loss, função de ativação sigmoide, 1 camada oculta de "n" neurônios, hiperparâmetro "h", máximo de iterações "maxit" e convergência usando a diferença sobre a função custo "conv" onde "n", "h", "maxit" e "conv" são parâmetros a serem definido pelo usuário, tendo valores definidos por padrão. Outro fator que pode ser definido pelo usuário é o limite inferior e superior da distribuição uniforme dos pesos e vieses (Ws e bs). Ainda que direcionado à porta XOR, garante a modelagem de outras portas lógicas conhecidas AND, NOR, OR, NAND, XNOR. Dependências: matplotlib, numpy, argparse*'
)

parser.add_argument(
    "-g",
    "--gate",
    type=str,
    default="XOR",
    metavar="porta lógica",
    help="insira a porta lógica desejada. Opções: AND, OR, NOR, NAND, XOR, XNOR. Default = XOR - Argumento do tipo: str",
)

parser.add_argument(
    "-n",
    "--nnum",
    type=int,
    default=4,
    metavar="neurônios",
    help="insira o número de neurônios presentes na camada oculta. Default = 4 - Argumento do tipo: int",
)

parser.add_argument(
    "--lr",
    type=float,
    default=1.0,
    metavar="aprendizado",
    help="insira a taxa de aprendizado (learning rate). Default = 1.0 - Argumento do tipo: float",
)


parser.add_argument(
    "--maxit",
    type=int,
    default=6000,
    metavar="máximo de iterações",
    help="insira o número máximo de iterações. Default = 5000 - Argumento do tipo: int",
)

parser.add_argument(
    "--conv",
    type=float,
    default=1e-08,
    metavar="convergência",
    help='insira o valor de convergência para a diferença entre a função custo em relaçao à função custo anterior, basicamente |C(n+1) - C(n)| < "conv". Default = 1e-08 - Argumento do tipo: float',
)

parser.add_argument(
    "--lim",
    type=float,
    default=[0, 0],
    metavar="convergência",
    nargs=2,
    help="insira o valor limite inferior e superior da distribuição uniforme dos pesos e vieses (Ws e bs). Default = 0 0. Por padrão, esses valores resultam na Inicialização de Pesos de Xavier Normalizada. Caso especial 1 1: Pesos e vieses sugeridos - Argumento do tipo: float float",
)


args = parser.parse_args()

eta = args.lr
neuron_num = args.nnum
max_it = args.maxit
convergence_crit = args.conv

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])


match args.gate:
    case "AND":
        y = np.array([0, 0, 0, 1])

    case "NOR":
        y = np.array([1, 0, 0, 0])

    case "OR":
        y = np.array([0, 1, 1, 1])

    case "NAND":
        y = np.array([1, 1, 1, 0])

    case "XOR":
        y = np.array([0, 1, 1, 0])

    case "XNOR":
        y = np.array([1, 0, 0, 1])

    case _:
        print(f"Porta {args.gate} Inválida")
        exit()


N = len(y)  # Número de elementos do array de entradae também saída


def activation_function(z):
    """
    Função de ativação (sigmoide)
    """
    return 1 / (1 + np.exp(-z))


def activation_function_prime(z):
    """
    Derivada da função de ativação (sigmoide)
    """
    return (1 - activation_function(z)) * activation_function(z)


def cost_function(y, Y):
    """
    Função Custo. Neste caso, a função log-loss somada sobre todos os elementos
    de Y e y e dividida pelo número de elementos
    """
    global N

    return 1 / N * np.sum((-y) * np.log(Y) - (1 - y) * np.log(1 - Y))


def grad_descent(y, X1, X2, W10, W20, b10, b20):
    """
    Gradiente Descendente. Composição de operações que minimizam a Função Custo
    """
    global eta, N, max_it, convergence_crit, neuron_num

    Wl1, Wl2 = W10, W20
    bl1, bl2 = b10, b20

    convergence = 1

    count = 0

    deltal1 = np.zeros((neuron_num, N))

    # Composição de x1 e x2 em um único array por conveniência de algumas
    # operações como é o caso de dWl1 (variação dos pesos da camada l = 1)
    x = np.array((X1, X2))

    while convergence > convergence_crit:
        """
        Aqui as operações são N-Dimensionais e não necessáriamente 1-D
        """

        Zl1 = np.zeros((neuron_num, N))
        for k in range(neuron_num):
            Zl1[k] = Wl1[k, 0] * X1 + Wl1[k, 1] * X2 + bl1[k]

        Yl1 = activation_function(Zl1)

        Zl2 = np.zeros(N)
        for i in range(neuron_num):
            Zl2 += Wl2[i] * Yl1[i] + bl2

        Yl2 = activation_function(Zl2)

        C0 = cost_function(y, Yl2)

        """
            Cálculo dos deltas, sendo delta(l=1) um array de dimensões k x 4 e
            delta(l=2) um array 1-D de 4 elementos
        """
        for k in range(neuron_num):
            deltal1[k] = Wl2[k] * (Yl2 - y) * activation_function_prime(Zl1[k])

        deltal2 = Yl2 - y

        """
            Atualização de Wl1, bl1, Wl2 e bl2 (Back-propagation)
        """
        dWl1 = -eta / N * deltal1.dot(x.T)
        Wl1 += dWl1

        dbl1 = -eta / N * np.sum(deltal1, axis=1)
        bl1 += dbl1

        dWl2 = -eta / N * deltal2.dot(Yl1.T)
        Wl2 += dWl2

        dbl2 = -eta / N * np.sum(deltal2)
        bl2 += dbl2

        """
            Para estabelecer uma relação de convergência, roda-se o passo 
            inicial do loop novamente, dessa vez, com valores atualizados de 
            Wl1, bl1, Wl2, bl2
        """
        Zl1 = np.zeros((neuron_num, N))
        for k in range(neuron_num):
            Zl1[k] = Wl1[k, 0] * X1 + Wl1[k, 1] * X2 + bl1[k]

        Yl1 = activation_function(Zl1)

        Zl2 = np.zeros(N)
        for i in range(neuron_num):
            Zl2 += Wl2[i] * Yl1[i] + bl2

        Yl2 = activation_function(Zl2)

        C1 = cost_function(y, Yl2)  # Função Custo atualizada -> C(n+1)

        """
        Teste do critério de convergência
        """
        convergence = abs(C1 - C0)

        count += 1

        if count == max_it:
            break

    return Wl1, Wl2, bl1, bl2, Yl2, count


"""
    Informações importantes:
        - A função numpy.random.rand() produz valores em uma distribuição
          uniforme com intervalo [0,1] que é equivalente a U(0,1)

        - Os pesos e biases devem, preferencialmente, seguir distribuições
          uniformes

    Por padrão, os pesos são inicializados através da Inicialização de Pesos de
    Xavier Normalizada

    U(-sqrt(6/(n + m)), sqrt(6/(n + m))) onde "n" é o número de componentes da
    camada anteiror e "m" o número de componentes da camada seguinte 
"""
match args.lim:
    case [0, 0]:  # Inicialização Padrão
        lowerl1 = -((6 / (2 + 1)) ** (1 / 2))
        upperl1 = (6 / (2 + 1)) ** (1 / 2)

        W10 = np.random.uniform(lowerl1, upperl1, size=(neuron_num, 2))
        b10 = np.zeros(neuron_num)

        lowerl2 = -((6 / (neuron_num + 0)) ** (1 / 2))
        upperl2 = (6 / (neuron_num + 0)) ** (1 / 2)

        W20 = np.random.uniform(lowerl2, upperl2, size=(neuron_num))
        b20 = 0

        distrib = " W ~ U(-sqrt(6/(n+m)),sqrt(6/(n+m)))"

    case _:  # Caso da escolha dos limites
        lower = args.lim[0]
        upper = args.lim[1]

        W10 = np.random.uniform(lower, upper, size=(neuron_num, 2))
        b10 = np.random.uniform(lower, upper, size=(neuron_num))

        W20 = np.random.uniform(lower, upper, size=(neuron_num))
        b20 = np.random.uniform(lower, upper, size=None)

        distrib = f" W ~ U({lower},{upper})"


print("Pesos (Weights) e Vieses (Biases) Iniciais" + distrib + ":\n")

print(f"W(l=1)\n{W10}\n\nb(l=1)\n{b10}\n")

print(f"W(l=2)\n{W20}\n\nb(l=2)\n{b20}\n")


"""
    Designação das variáveis relativas às quantidades a serem obtidas a partir
    da back-propagation utilizando o método do gradiente descendente 
"""
Wl1, Wl2, bl1, bl2, Yl2, count = grad_descent(y, x1, x2, W10, W20, b10, b20)


print(f"\nTaxa de Aprendizado: {eta}")

print(f"Neurônios da Camada Oculta: {neuron_num}")

print(f"Número de Iterações: {count} de {max_it}")

print(f"Critério de Convergência: |C(n+1) - C(n)| < {convergence_crit}")

print("\nTabela de Classificação:\n")

s = np.array(range(1, N + 1))

print("      s  |  X1  |  X2  |  y  |     Y")
print("     -----------------------------------")

for i in range(N):
    print(f"      {s[i]}  |   {x1[i]}  |   {x2[i]}  |  {y[i]}  |  {round(Yl2[i],5)}")


def predict(x1, x2, Wl1=Wl1, Wl2=Wl2, bl1=bl1, bl2=bl2):
    """
    Faz as predições de acordo com os pesos e "biases" obtidos
    anteriormente

    O uso da função flatten permite que arrays multidimensionais sejam
    utilizados como input para x1 e x2, convenientemente, para plotar
    gráficos do tipo contour
    """

    # Operações da primeira camada. l = 1
    Zl1 = np.zeros((len(bl1), len(x1.flatten())))
    for k in range(len(bl1)):
        Zl1[k] = Wl1[k, 0] * x1.flatten() + Wl1[k, 1] * x2.flatten() + bl1[k]

    Yl1 = activation_function(Zl1)

    # Operações da segunda camada. l = 2
    Zl2 = np.zeros(len(x1.flatten()))
    for k in range(len(bl1)):
        Zl2 += Wl2[k] * Yl1[k] + bl2

    Yl2 = activation_function(Zl2)

    return Yl2  # Yl2 = Yout


# Arrays multidimensionais de pontos variados para gerar combinações de x1 e x2
# a serem calculados e mostrados
x1print, x2print = np.meshgrid(np.arange(0.1, 1, 0.2), np.arange(0.1, 1, 0.2))

# Previsão dos valores de Y para x1print e x2print + restruturação em forma de
# array de mesma dimensão de x1print e x2print
Yprint = predict(x1print, x2print).reshape(x1print.shape)


"""
    GRÁFICOS
"""

# Arrays multidimensionais de pontos variados para gerar combinações de x1 e x2
# a serem calculados e plotados em um gráfico do tipo contour
x1plot, x2plot = np.meshgrid(np.arange(0.0, 1.01, 0.01), np.arange(0.0, 1.01, 0.01))

# Previsão dos valores de Y para x1plot e x2plot + restruturação em forma de
# array de mesma dimensão de x2plot e x2plot
Yplot = predict(x1plot, x2plot).reshape(x1plot.shape)

fig1, ax = plt.subplots(constrained_layout=True)
CS = ax.contourf(x1plot, x2plot, Yplot, 10)
fig1.colorbar(CS, ax=ax, label="$Y(X_1,X_2)$")
ax.set_xlabel("$X1$")
ax.set_ylabel("$X2$")
ax.set_title("Porta " + f"{args.gate}" + " | Back-propagation")

plt.show()
