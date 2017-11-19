int led = 13;
int enable = 8;
int inputA = 9;
int inputB = 10;
int pwm = 11;
int flag = 0;


// the setup routine runs once when you press reset:
void setup() {                
  // initialize the digital pin as an output.
  pinMode(led, OUTPUT); 
  pinMode(enable, OUTPUT);  
  pinMode(inputA, OUTPUT);  
  pinMode(inputB, OUTPUT);  
  pinMode(pwm, OUTPUT); 
  Serial.begin(9600); 
}

// the loop routine runs over and over again forever:
void loop() {
  if(Serial.available() > 0)
  {
    if(Serial.read() == '1' )
    {
      flag = 1;
    }
  }  

  if (flag == 1){
  digitalWrite(enable, HIGH);   
  digitalWrite(inputA, LOW); 
  digitalWrite(inputB, HIGH);
  digitalWrite(led, HIGH);
  analogWrite(pwm, 255);    
  delay(60);
  digitalWrite(enable, LOW);
  digitalWrite(led, LOW);
  flag = 0;
  }
  
}


