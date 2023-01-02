import cheatahh.nn.LayerCache
import cheatahh.nn.NeuralNetworkFP32
import kotlin.random.Random
import kotlin.system.measureTimeMillis

const val markerPass = "\u001B[92m✓\u001B[0m"
const val markerFail = "\u001B[91m✗\u001B[0m"

fun main() {

    fun test(a: Boolean, b: Boolean) {
        val (result, certainty) = XOR(a, b)
        val expectedResult = a xor b
        println("${if(result == expectedResult) markerPass else markerFail} ($a, $b) = $result @${certainty * 100}%, expected = $expectedResult")
    }

    println("Before training")
    test(a = false, b = false)
    test(a = false, b = true)
    test(a = true, b = false)
    test(a = true, b = true)

    println("Training finished in ${
        measureTimeMillis {
            XOR.doTraining(10000)
        }
    }ms")

    println("After training")
    test(a = false, b = false)
    test(a = false, b = true)
    test(a = true, b = false)
    test(a = true, b = true)
}

object XOR : NeuralNetworkFP32(2, 5, 1) {

    private val inputCache = FloatArray(2)
    private val targetCache = FloatArray(1)

    operator fun invoke(a: Boolean, b: Boolean) : Pair<Boolean, Float> {
        inputCache[0] = boolean2Float(a)
        inputCache[1] = boolean2Float(b)
        val result = feed(inputCache)
        return if(result[0] >= 0.5f) {
            true to result[0]
        } else {
            false to 1 - result[0]
        }
    }

    private fun train(a: Boolean, b: Boolean, target: Boolean, cache: LayerCache<FloatArray>) {
        inputCache[0] = boolean2Float(a)
        inputCache[1] = boolean2Float(b)
        targetCache[0] = boolean2Float(target)
        train(inputCache, targetCache, cache)
    }

    fun doTraining(rounds: Int) {
        val random = Random
        val cache = newLayerCache()
        repeat(rounds) {
            val a = random.nextBoolean()
            val b = random.nextBoolean()
            train(a, b, a xor b, cache)
        }
    }

    private fun boolean2Float(z: Boolean) = if(z) 1f else 0f
}