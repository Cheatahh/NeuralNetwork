@file:Suppress("DuplicatedCode", "MemberVisibilityCanBePrivate", "unused")

package cheatahh.nn

import kotlin.math.exp

fun sigmoid(value: Float) = 1f / (1 + exp(-value))
fun derive(value: Float) = value * (1 - value)

open class NeuralNetworkFP32(
    vararg layerSizes: Int,
    weightInitializer: WeightInitializer<FloatArray> = WeightInitializer.randomFP32,
    biasInitializer: BiasInitializer<FloatArray> = BiasInitializer.randomFP32
) : NeuralNetwork<FloatArray> {

    init {
        require(layerSizes.size >= 2) { "At least 2 layers are required (input & output)" }
    }

    override val weights = Array(layerSizes.size - 1) { layer ->
        Array(layerSizes[layer + 1]) { targetNeuron ->
            FloatArray(layerSizes[layer]).apply {
                weightInitializer(layer + 1, targetNeuron, this)
            }
        }
    }

    override val biases = Array(layerSizes.size - 1) { layer ->
        FloatArray(layerSizes[layer + 1]).apply {
            biasInitializer(layer + 1, this)
        }
    }

    override fun newLayerCache() = LayerCache(weights.size) {
        FloatArray(weights[it].size)
    }

    override fun feed(input: FloatArray, layerCache: LayerCache<FloatArray>): FloatArray {
        for (layerIndex in layerCache.indices) {
            val layer = layerCache[layerIndex]
            val prevLayer = if(layerIndex == 0) input else layerCache[layerIndex - 1]
            val layerWeights = weights[layerIndex]
            val layerBiases = biases[layerIndex]
            layerWeights.forEachIndexed { targetNeuron, weights ->
                var brightness = 0f
                for (weightIndex in weights.indices)
                    brightness += prevLayer[weightIndex] * weights[weightIndex]
                layer[targetNeuron] = sigmoid(brightness + layerBiases[targetNeuron])
            }
        }
        return layerCache.last()
    }

    override fun train(input: FloatArray, target: FloatArray, layerCache: LayerCache<FloatArray>) =
        train(input, target, layerCache, NeuralNetwork.DEFAULT_LEARNING_RATE)

    fun train(input: FloatArray, target: FloatArray, layerCache: LayerCache<FloatArray>, learningRate: Float) {
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
                var error = 0f
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