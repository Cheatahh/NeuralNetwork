@file:Suppress("DuplicatedCode", "MemberVisibilityCanBePrivate", "unused")

package cheatahh.nn

import kotlin.math.exp

fun sigmoid(value: Double) = 1.0 / (1 + exp(-value))
fun derive(value: Double) = value * (1 - value)

open class NeuralNetworkFP64(
    vararg layerSizes: Int,
    weightInitializer: WeightInitializer<DoubleArray> = WeightInitializer.randomFP64,
    biasInitializer: BiasInitializer<DoubleArray> = BiasInitializer.randomFP64
) : NeuralNetwork<DoubleArray> {

    init {
        require(layerSizes.size >= 2) { "At least 2 layers are required (input & output)" }
    }

    override val weights = Array(layerSizes.size - 1) { layer ->
        Array(layerSizes[layer + 1]) { targetNeuron ->
            DoubleArray(layerSizes[layer]).apply {
                weightInitializer(layer + 1, targetNeuron, this)
            }
        }
    }

    override val biases = Array(layerSizes.size - 1) { layer ->
        DoubleArray(layerSizes[layer + 1]).apply {
            biasInitializer(layer + 1, this)
        }
    }

    override fun newLayerCache() = LayerCache(weights.size) {
        DoubleArray(weights[it].size)
    }

    override fun feed(input: DoubleArray, layerCache: LayerCache<DoubleArray>): DoubleArray {
        for (layerIndex in layerCache.indices) {
            val layer = layerCache[layerIndex]
            val prevLayer = if(layerIndex == 0) input else layerCache[layerIndex - 1]
            val layerWeights = weights[layerIndex]
            val layerBiases = biases[layerIndex]
            layerWeights.forEachIndexed { targetNeuron, weights ->
                var brightness = 0.0
                for (weightIndex in weights.indices)
                    brightness += prevLayer[weightIndex] * weights[weightIndex]
                layer[targetNeuron] = sigmoid(brightness + layerBiases[targetNeuron])
            }
        }
        return layerCache.last()
    }

    override fun train(input: DoubleArray, target: DoubleArray, layerCache: LayerCache<DoubleArray>) =
        train(input, target, layerCache, NeuralNetwork.DEFAULT_LEARNING_RATE.toDouble())

    fun train(input: DoubleArray, target: DoubleArray, layerCache: LayerCache<DoubleArray>, learningRate: Double) {
        feed(input, layerCache)

        fun evalOutputLayerError() {
            val layer = layerCache.last()
            for (index in layer.indices) {
                val layerOutput = layer[index]
                layer[index] = -(target[index] - layerOutput) * derive(layerOutput)
            }
        }

        fun evalHiddenLayerError(layerIndex: Int) {
            val layer = layerCache[layerIndex]
            val nextLayerError = layerCache[layerIndex + 1]
            val nextLayerWeights = weights[layerIndex + 1]
            for (index in layer.indices) {
                val layerOutput = layer[index]
                var error = 0.0
                for (o in nextLayerError.indices)
                    error += nextLayerError[o] * nextLayerWeights[o][index]
                layer[index] = error * derive(layerOutput)
            }
        }

        fun adjustLayerWeights(layerIndex: Int) {
            val error = layerCache[layerIndex]
            val layerWeights = weights[layerIndex]
            val prevLayer = if (layerIndex == 0) input else layerCache[layerIndex - 1]
            for (o in error.indices) {
                val neuronWeights = layerWeights[o]
                for (wh in neuronWeights.indices)
                    neuronWeights[wh] -= learningRate * error[o] * prevLayer[wh]
            }
        }

        evalOutputLayerError()
        adjustLayerWeights(layerCache.size - 1)
        for (layerIndex in layerCache.size - 2 downTo 0) {
            evalHiddenLayerError(layerIndex)
            adjustLayerWeights(layerIndex)
        }
    }
}