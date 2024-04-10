float inputValue = 0.0; // Used to store the floating-point number received from the Raspberry Pi
int outputValue = 0; // Used to store the converted analog output value

void setup() {
  Serial.begin(115200); // Start serial communication with a baud rate of 115200
  pinMode(A0, OUTPUT); // Set pin A0 as an output
}

void loop() {
  if (Serial.available() > 0) { // Check if there is data available to read from the serial port
    inputValue = Serial.parseFloat(); // Read serial data and convert it to a floating-point number
    if (Serial.read() == '\n') { // Check for the end-of-data marker (newline character)
      outputValue = mapFloatToPWM(inputValue); // Map the floating-point number to a PWM value
      analogWrite(A0, outputValue); // Output the PWM signal to pin A0
    }
  }

  int analogInput1 = analogRead(A1); // Read the analog signal from pin A1
  int analogInput2 = analogRead(A2); // Read the analog signal from pin A2
  // Send the analog signal values from pins A1 and A2 back to the Raspberry Pi via serial, separated by a comma
  Serial.print(analogInput1);
  Serial.print(",");
  Serial.println(analogInput2);

  delay(10); // Simple delay to reduce data volume
}

// Custom function to map a floating-point number to the PWM output range (0-255)
// Assuming the input value range is -1.0 to 1.0
int mapFloatToPWM(float value) {
  return map(value * 100, -100, 100, 0, 255);
}
