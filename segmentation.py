import cv2

# 全局變數，用於儲存滑鼠事件的起始和結束點
ref_point = []
cropping = False

# 定義滑鼠回呼函數，用於處理滑鼠事件
def mouse_crop(event, x, y, flags, param):
    global ref_point, cropping
    
    # 滑鼠左鍵按下時，記錄起始點
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # 滑鼠拖動時，動態顯示選取的範圍
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp_image = image.copy()
        cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", temp_image)
        
    # 滑鼠左鍵放開時，記錄結束點，並顯示選取範圍的大小
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        
        # 在圖片上畫出選取的矩形
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

        # 計算選取區域的寬度和高度
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        print(f"選取範圍的大小：寬度 {width}, 高度 {height}")

# 讀取圖片
image = cv2.imread('IMG_5977.JPG')
clone = image.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

# 顯示圖片，等待用戶選取區域
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # 按下 "r" 重置選取
    if key == ord("r"):
        image = clone.copy()

    # 按下 "q" 結束程式
    elif key == ord("q"):
        break

cv2.destroyAllWindows()



# image size: 300 * 300
