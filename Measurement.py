class Measurement:
    
    def __init__(self, accelerometer, accelero_link,  gyroscope, gyro_link):
        self.accelerometer = accelerometer
        self.accelero_link = accelero_link
        self.gyroscope = gyroscope
        self.gyro_link = gyro_link
        pass
    
    # #GET
    # def _get_accelerometer(self):
    #     return self.accelerometer
    # def _get_accelero_link(self):
    #     return self.accelero_link
    # def _get_gyroscope(self):
    #     return self.gyroscope 
    # def _get_gyro_link(self):
    #     return self.gyro_link
    
    # #SET
    # def _set_accelerometer(self, value):
    #     self.accelerometer = value
    # def _set_accelero_link(self, value):
    #     self.accelero_link = value
    # def _set_gyroscope(self, value):
    #     self.gyroscope = value
    # def _set_gyro_link(self, value):
    #     self.gyro_link = value
        
    #Creating the properties I guess
    # accelerometer = property(fget=_get_accelerometer, fset=_set_accelerometer)
    # accelero_link = property(fget=_get_accelero_link, fset=_set_accelero_link)
    # gyroscope = property(fget=_get_gyroscope, fset=_set_gyroscope)
    # gyro_link = property(fget=_get_gyro_link, fset=_set_gyro_link)