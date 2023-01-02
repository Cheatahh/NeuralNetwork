plugins {
    kotlin("multiplatform") version "1.8.0"
}

group = "cheatahh.nn"
version = "0.1"

repositories {
    mavenCentral()
}

kotlin {
    jvm("jvm")
    mingwX64("native") {
        binaries {
            executable {
                entryPoint = "main"
            }
        }
    }
}
