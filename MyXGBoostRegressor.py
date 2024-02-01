{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP3eWigwsrL9hfYOehgO4K7",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Jong-Chan-Lee/data/blob/main/MyXGBoostRegressor.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "K1A2hEDCs9ei"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from collections import Counter\n",
        "import copy"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class MyXGBRegressionTree:\n",
        "  def __init__(self, max_depth, reg_lambda, prune_gamma):\n",
        "    self.max_depth = max_depth\n",
        "    self.reg_lambda = reg_lambda\n",
        "    self.prune_gamma = prune_gamma\n",
        "    self.estimator1 = None\n",
        "    self.estimator2 = None\n",
        "    self.feature = None\n",
        "    self.residual = None\n",
        "    self.base_score = None\n",
        "\n",
        "  def node_split(self, did):\n",
        "    r = self.reg_lambda\n",
        "    max_gain = -np.inf\n",
        "    d = self.feature.shape[1]\n",
        "    G = -self.residual[did].sum()\n",
        "    H = did.shape[0]\n",
        "    p_score = (G ** 2) / (H + r)\n",
        "\n",
        "    for k in range(d):\n",
        "      GL = HL = 0.0\n",
        "      x_feat = self.feature[did, k]\n",
        "      x_inoq = np.unique(x_feat)\n",
        "      s_point = [np.mean([x_uniq[i-1], x_uniq[i]]) \\\n",
        "                 for i in range(1, len(x_uniq))]\n",
        "      l_bound = -np.inf\n",
        "      for j in s_point:\n",
        "        left = did[np.where(np.logical_and(x_feat > l_bound, x_feat <= j))[0]]\n",
        "        right = did[np.where(x_feat > j)[0]]\n",
        "        GL -= self.residual[left].sum()\n",
        "        HL += left.shape[0]\n",
        "        GR = G - GL\n",
        "        HR = H - HL\n",
        "        gain = (GL ** 2) / (HR + r) + (GR ** 2) / (HR + r) - p_score\n",
        "        if gain > max_gain:\n",
        "          max_gain = gain\n",
        "          b_fid = k\n",
        "          b_point = j\n",
        "        l_bound = j\n",
        "    if max_gain >= self.prune_gamma:\n",
        "      x_feat = self.feature[did, b_fid]\n",
        "      b_left = did[np.where(x_feat <= b_point)[0]]\n",
        "      b_right = did[np.where(x_feat > b_point)[0]]\n",
        "      return {'fid':b_fid, 'split_point':b_point, 'left':b_left, 'right':b_right}\n",
        "    else :\n",
        "      return np.nan"
      ],
      "metadata": {
        "id": "zXxSJwUFtc1r"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "  def recursive_splite(self, node, curr_depth):\n",
        "    left = node['left']\n",
        "    right = node['right']\n",
        "    if curr_depth >= selp.max_depth:\n",
        "      return\n",
        "\n",
        "    s = self.node_split(left)\n",
        "    if isinstance(s, dict):\n",
        "      node['left'] = s\n",
        "      self.recursive_split(node['left'], curr_depth+1)\n",
        "    s =  self.node_split(right)\n",
        "    if isinstance(s, dict):\n",
        "      node['right'] = s\n",
        "      self.recursive_split(node['right'], curr_depth+1)\n",
        "\n",
        "  def output_value(self, did):\n",
        "    r = self.residual[did]\n",
        "    return np.sum(r) / (did.shape[0] + self.reg_lambda)\n",
        "\n",
        "  def output_leaf(self, d):\n",
        "    if isinstance(d, dict):\n",
        "      for key, value in d.items():\n",
        "        if key == 'left' or key == 'right':\n",
        "          rtn = self.output_leaf(value)\n",
        "          if rtn[0] == 1:\n",
        "            d[key] = rtn[1]\n",
        "      return 0, 0\n",
        "    else:\n",
        "      return 1, self.output_value(d)\n",
        "\n",
        "  def fit(self, x, y):\n",
        "    self.feature = x\n",
        "    self.residual = y\n",
        "    self.base_score = y.mean()\n",
        "\n",
        "    root = self.node_split(np.arange(x.shape[0]))\n",
        "    if isinstance(root, dict):\n",
        "      self.recursive(root, curr_depth=1)\n",
        "\n",
        "    self.estimator1 = np.root\n",
        "\n",
        "    if isinstance(self.estimator1, dict):\n",
        "      self.estimator2 = copy.deepcopy(self.estimator1)\n",
        "      self.output_leaf(self.estimator2)\n",
        "    return self.estimator2"
      ],
      "metadata": {
        "id": "CWnahcr4vroI"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "  def x_predict(self, p, x):\n",
        "    if x[p['fid']] <= p['split_point']:\n",
        "      if isinstance(p['left'], dict):\n",
        "        return self.x_predict(p['left'], x)\n",
        "      else:\n",
        "        return p['left']\n",
        "    else:\n",
        "      if isinstance(p['right'], dict):\n",
        "        return self.x_predict(p['right'], x)\n",
        "      else:\n",
        "        return p['right']\n",
        "\n",
        "  def predict(self, x_test):\n",
        "    p = self.estimator2\n",
        "    if isinstance(p, dict):\n",
        "      y_pred = [self.x_predict(p,x) for x in x_test]\n",
        "      return np.array(y_pred)\n",
        "    else:\n",
        "      return np.array([self.base_score] * x_test.shape[0])\n",
        "\n",
        "class MyXGBRegressor:\n",
        "    def __init__(self, n_estimators=10, max_depth=3,\n",
        "               learning_rate=0.3, prune_gamma=0.0,\n",
        "               reg_lambda=0.0, base_score=0.5):\n",
        "      self.n_estimators = n_estimators\n",
        "      self.max_depth = max_depth\n",
        "      self.eta = learning_rate\n",
        "      self.prune_gamma = prune_gamma\n",
        "      self.reg_lambda = reg_lambda\n",
        "      self.base_score = base_score\n",
        "      self.estimator1 = dict()\n",
        "      self.estimator2 = dict()\n",
        "      self.models = []\n",
        "      self.loss = []\n",
        "\n",
        "    def fit(self, x, y):\n",
        "      Fm = self.base_score\n",
        "      self,models = []\n",
        "      self.loss = []\n",
        "      for m in range(self.n_estimators):\n",
        "        residual = y - Fm\n",
        "        model = MyXGBRegressionTree(max_depth=self.max_depth,\n",
        "                                reg_lambda=self.reg_lambda,\n",
        "                                prune_gamma = self.prune_gamma)\n",
        "        model.fit(x, residual)\n",
        "        gamma = model.predict(x)\n",
        "        Fm = Fm + self.eta * gamma\n",
        "        self.models.append(model)\n",
        "        self.loss.append(((y - Fm) ** 2).sum())\n",
        "      return self.loss\n",
        "\n",
        "    def predict(self, x_test):\n",
        "      y_pred = np.zeros(shape=(x_test.shape[0],)) + self.base_score\n",
        "      for model in self.models:\n",
        "        y_pred += self.eta * model.predict(x_test)\n",
        "      return y_pred"
      ],
      "metadata": {
        "id": "C7cNQwKxgMXL"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from MyXGBoostRegressor import MyXGBRegressor\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def nonlinear_data(n,s):\n",
        "  rtn_x, rtn_y = [], []\n",
        "  for i in ranfe(n):\n",
        "    x = np.random.random()\n",
        "    y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, s) + 3.0\n",
        "    rtn_x.append(x)\n",
        "    rtn_y.append(y)\n",
        "  return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)\n",
        "x,y = nonlinear_data(n=500, s=0.5)\n",
        "\n",
        "def plot_prediction(x, y, x_test, y_pred):\n",
        "    plt.figure(figsize=(5,4))\n",
        "    plt.scatter(x,y,c='blue',s=20,alpha=0.5,label='train_data')\n",
        "    plt.plot(x_test, y_pred, c='red', lw=2.0, label='prediction')\n",
        "    plt.xlim(0,1)\n",
        "    plt.tlim(0,7)\n",
        "    plt.legend()\n",
        "    plt.show()\n",
        "\n",
        "y_mean = y.mean()\n",
        "n_depth = 3\n",
        "n_subtree = 20\n",
        "eta = 0.3\n",
        "reg_lambda = 1.0\n",
        "prune_gamma = 2.0\n",
        "my_model = MyXGBRegressor(n_estimators=n_subtree,\n",
        "                          max_depth=n_depth,\n",
        "                          learning_rate=eta,\n",
        "                          prune_gamma = prune_gamma,\n",
        "                          reg_lambda = reg_lambda,\n",
        "                          base_score = y_mean)\n",
        "loss = my_model.fit(x,y)\n",
        "\n",
        "plt.figure(figsize=(5.4))\n",
        "plt.plot(loss, c='red')\n",
        "plt.xlabel('m : iteration')\n",
        "plt.ylabel('loss : mean squared error')\n",
        "plt.title('loss history')\n",
        "plt.show()\n",
        "\n",
        "x_test = np.linspace(0,1,50).reshape(-1,1)\n",
        "y_pred = my_model.predict(x_test)\n",
        "plot_prediction(x,y,x_test,y_pred)\n",
        "\n",
        "from xgboost import XGBRegressor\n",
        "xg_model = XGBRegressor(n_estimators=n_subtree,\n",
        "                        max_depth=n_depth,\n",
        "                        learning_rate = eta,\n",
        "                        gamma=prune_gamma,\n",
        "                        reg_lambda=reg_lambda,\n",
        "                        base_score=y_mean)\n",
        "xg_model.fit(x,y)\n",
        "y_pred=xg_model.predict(x_test)\n",
        "plot_prediction(x,y,x_test,y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 404
        },
        "id": "HsDHU2ExyxiO",
        "outputId": "effca7f2-7eb4-4a26-b1c1-ca65c59f9b03"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'MyXGBoostRegressor'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-7-e30d61e773ca>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mMyXGBoostRegressor\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mMyXGBRegressor\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmatplotlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpyplot\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mnonlinear_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mn\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'MyXGBoostRegressor'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ]
    }
  ]
}