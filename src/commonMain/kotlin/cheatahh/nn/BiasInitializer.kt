package cheatahh.nn

import kotlin.random.Random

fun interface BiasInitializer<PrecisionSpec> {

    operator fun invoke(targetLayer: Int, targetBiases: PrecisionSpec)

    companion object {
        val randomFP32 = Random.let { random ->
            BiasInitializer<FloatArray> { _, biases ->
                for(index in biases.indices) biases[index] = random.nextFloat()
            }
        }
        val randomFP64 = Random.let { random ->
            BiasInitializer<DoubleArray> { _, biases ->
                for(index in biases.indices) biases[index] = random.nextDouble()
            }
        }
    }
}