# NeuralNetwork

Simple neural network implementation written in Kotlin.

### Example Usage:

See `src/*Main/kotlin/XOR.kt` for a detailed example.

```kotlin
val network = NeuralNetworkFP32(2, 5, 1)

val testData = listOf(
    floatArrayOf(0f, 0f) to floatArrayOf(0f),
    floatArrayOf(1f, 0f) to floatArrayOf(1f),
    floatArrayOf(0f, 1f) to floatArrayOf(1f),
    floatArrayOf(1f, 1f) to floatArrayOf(0f)
)

// incorrect results, ~usually 50% misses
for((inputs, _) in testData) {
    println(network(inputs).contentToString())
}

network.train(testData, 10000)

// correct results
for((inputs, _) in testData) {
    println(network(inputs).contentToString())
}
```
