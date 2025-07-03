import cv2
import base64
from openai import OpenAI

client = OpenAI(api_key="")
def encode_image_from_array(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode("utf-8")

def capture_and_describe():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press SPACE to capture a frame and send to GPT-4 Vision.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam (Press SPACE to capture)", frame)

        key = cv2.waitKey(1)
        if key % 256 == ord('q'):
            break
        elif key % 256 == 32:  # SPACE pressed
            print("Capturing frame...")
            base64_image = encode_image_from_array(frame)

            print("Sending image to GPT-4 Vision...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert computer vision model describing images."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe everything you see in this image in detail."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            description = response.choices[0].message.content
            print("\n=== GPT-4 Vision Response ===")
            print(description)
            print("\nPress SPACE to capture again or Q to quit.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_describe()
