
import cv2

class VehicleCounter:
    def __init__(self, line_position=2 / 3, margin=10):
        self.line_position = line_position
        self.margin = margin
        self.tracked_objects = {}
        self.count_down = 0
        self.count_up = 0
        self.line_y = None

    def set_line_position(self, height):
        """Calcula la posición Y de la línea basada en la altura del frame"""
        self.line_y = int(height * self.line_position)

    def update(self, track_id, center_y):
        """Actualiza el tracking y cuenta si cruza la línea"""
        if self.line_y is None:
            return

        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = {'last_y': center_y, 'counted': False}
        else:
            last_y = self.tracked_objects[track_id]['last_y']

            if not self.tracked_objects[track_id]['counted']:
                # Detecta cruce interpolando entre last_y y center_y. Realmente solo mira la diferencia de el anterior ¡centroide del car contra el actual
                if last_y < self.line_y and center_y > self.line_y:
                    self.count_down += 1
                    self.tracked_objects[track_id]['counted'] = True
                elif last_y > self.line_y and center_y < self.line_y:
                    self.count_up += 1
                    self.tracked_objects[track_id]['counted'] = True

            self.tracked_objects[track_id]['last_y'] = center_y

    def draw(self, frame):
        """Dibuja la línea y los contadores en el frame"""
        if self.line_y is None:
            return

        # Línea horizontal
        width = frame.shape[1]
        cv2.line(frame, (0, self.line_y), (width, self.line_y), (0, 255, 0), 3)

        # Contadores
        cv2.putText(frame, f'Arriba->Abajo: {self.count_down}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Abajo->Arriba: {self.count_up}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    def draw_vertical(self,frame): #Esta es para después, ya la pondremos bien :)
        """Dibuja la línea y los contadores en el frame"""
        if self.line_y is None:
            return

        # Línea horizontal
        height = frame.shape[0]
        cv2.line(frame, (self.line_y,0), (self.line_y,height), (0, 255, 0), 3)

        # Contadores
        cv2.putText(frame, f'Izquierda->Derecha: {self.count_down}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Derecha->Izquierda: {self.count_up}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)