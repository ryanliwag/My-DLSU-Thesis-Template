int led = 13;
int enable = 12;
int inputA = 11;
int inputB = 10;
int pwm = 9;



// the setup routine runs once when you press reset:
void setup() {                
  // initialize the digital pin as an output.
  pinMode(led, OUTPUT); 
  pinMode(enable, OUTPUT);  
  pinMode(inputA, OUTPUT);  
  pinMode(inputB, OUTPUT);  
  pinMode(pwm, OUTPUT);  
}

// the loop routine runs over and over again forever:
void loop() {
  
  delay(1000);  
  digitalWrite(enable, HIGH);   
  digitalWrite(inputA, HIGH); 
  digitalWrite(inputB, LOW);
  analogWrite(pwm, 255);    
  delay(5000);
  digitalWrite(enable, LOW);
}
