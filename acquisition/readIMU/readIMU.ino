// Run on Arduino Giga R1
// IMU should have:
//      blue wire into SDA   (pin 20)
//      yellow wire into SCL (pin 21)
//      red is 3.3 V power
//      black is Gnd
//
// Uses sparkfun BMI270 library:
// https://github.com/sparkfun/SparkFun_BMI270_Arduino_Library
// Written by DMM, Aug 2025


#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"

BMI270 imu;

uint8_t i2cAddress = BMI2_I2C_PRIM_ADDR;

void setup() {

    Serial.begin(115200);
    Serial.println("BMI270 Example 1 - Basic Readings I2C");

    Wire.begin();

    // Check if sensor is connected and initialize
    while(imu.beginI2C(i2cAddress) != BMI2_OK) {
        Serial.println("Error: BMI270 not connected, check wiring and I2C address!");
        delay(1000);
    }

    Serial.println("BMI270 connected");

}

void loop() {

    imu.getSensorData();

    // Acceleration in g's
    Serial.print(imu.data.accelX, 3);
    Serial.print(",");
    Serial.print(imu.data.accelY, 3);
    Serial.print(",");
    Serial.print(imu.data.accelZ, 3);
    Serial.print(",");

    // Rotation in deg/sec
    Serial.print(imu.data.gyroX, 3);
    Serial.print(",");
    Serial.print(imu.data.gyroY, 3);
    Serial.print(",");
    Serial.println(imu.data.gyroZ, 3);

    // Print 50x per second
    delay(20);
}