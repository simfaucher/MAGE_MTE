class MTEThreshold:
    def __init__(self, number_keypoints: int, number_matches: int, mean_kirsh: int, \
    mean_canny: int, mean_color: int, kirsh_aberration: int, color_aberration: int):
        self.nb_kp = number_keypoints
        self.nb_match = number_matches
        self.mean_kirsh = mean_kirsh
        self.mean_canny = mean_canny
        self.mean_color = mean_color
        self.kirsh_aberration = kirsh_aberration
        self.color_aberration = color_aberration
