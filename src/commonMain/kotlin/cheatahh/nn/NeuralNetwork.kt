@file:Suppress("unused")

package cheatahh.nn

typealias LayerCache<PrecisionSpec> = Array<PrecisionSpec>

interface NeuralNetwork<PrecisionSpec> {

    val weights: Array<Array<PrecisionSpec>>
    val biases: Array<PrecisionSpec>

    fun newLayerCache(): LayerCache<PrecisionSpec>

    fun feed(input: PrecisionSpec, layerCache: LayerCache<PrecisionSpec> = newLayerCache()): PrecisionSpec
    fun train(input: PrecisionSpec, target: PrecisionSpec, layerCache: LayerCache<PrecisionSpec> = newLayerCache())

    fun train(dataset: List<Pair<PrecisionSpec, PrecisionSpec>>, iterations: Int, layerCache: LayerCache<PrecisionSpec> = newLayerCache()) {
        repeat(iterations) {
            val (input, target) = dataset.random()
            train(input, target, layerCache)
        }
    }

    operator fun invoke(input: PrecisionSpec, layerCache: LayerCache<PrecisionSpec> = newLayerCache()) = feed(input, layerCache)
    operator fun invoke(input: PrecisionSpec, target: PrecisionSpec, layerCache: LayerCache<PrecisionSpec> = newLayerCache()) = train(input, target, layerCache)

    companion object {
        const val DEFAULT_LEARNING_RATE = 0.5f
    }
}