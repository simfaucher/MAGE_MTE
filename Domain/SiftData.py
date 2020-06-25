class SiftData:
    def __init__(self, kp_380, desc_380, ref, kp_base, kp_640, desc_640, kp_1730, desc_1730):
        self.kp = kp_380
        self.des = desc_380
        self.current_size = 380
        self.ref = ref
        self.kp_base = kp_base
        self.kp_380 = kp_380
        self.kp_640 = kp_640
        self.kp_1730 = kp_1730
        self.desc_380 = desc_380
        self.desc_640 = desc_640
        self.desc_1730 = desc_1730

    def switch_380(self):
        if not self.current_size == 380:
            print("Switch 380")
            self.kp = self.kp_380
            self.des = self.desc_380
            self.current_size = 380

    def switch_640(self):
        if not self.current_size == 640:
            print("Switch 640")
            self.kp = self.kp_640
            self.des = self.desc_640
            self.current_size = 640

    def switch_1730(self):
        if not self.current_size == 1730:
            self.kp = self.kp_1730
            self.des = self.desc_1730
            self.current_size = 1730
        