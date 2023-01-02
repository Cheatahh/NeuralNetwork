package cheatahh.nn

import kotlin.random.Random

fun interface WeightInitializer<PrecisionSpec> {
    
    operator fun invoke(targetLayer: Int, targetNeuron: Int, baseNeuronToTargetWeights: PrecisionSpec)
    
    companion object {
        val randomFP32 = Random.let { random ->
            WeightInitializer<FloatArray> { _, _, weights ->
                for(index in weights.indices) weights[index] = random.nextFloat()
            }
        }
        val randomFP64 = Random.let { random ->
            WeightInitializer<DoubleArray> { _, _, weights ->
                for(index in weights.indices) weights[index] = random.nextDouble()
            }
        }
    }
}
