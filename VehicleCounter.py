import cv2

class VehicleCounter:
    def __init__(self, line_position=0.5, margin=10, min_movement=1.0, orientation='horizontal', direction=None):
        """
        Args:
            line_position: Posición relativa de la línea (0.0 a 1.0)
            margin: Tolerancia en píxeles
            min_movement: Umbral mínimo de movimiento
            orientation: 'horizontal' o 'vertical'
            direction: Para vertical: 'left_to_right' o 'right_to_left' (None = ambas direcciones)
        """
        self.line_position = line_position
        self.margin = margin
        self.min_movement = float(min_movement)
        self.orientation = orientation
        self.direction = direction
        self.tracked_objects = {}
        self.count_forward = 0   # arriba->abajo o izq->der
        self.count_backward = 0  # abajo->arriba o der->izq
        self.line_pos = None

    def set_line_position(self, height, width=None):
        """Calcula la posición de la línea según la orientación."""
        if self.orientation == 'horizontal':
            self.line_pos = float(height * self.line_position)
        else:  # vertical
            if width is None:
                raise ValueError("width requerido para orientación vertical")
            self.line_pos = float(width * self.line_position)

    def update(self, track_id, center_x=None, center_y=None):
        """Actualiza el tracking y cuenta cruces según orientación y dirección."""
        if self.line_pos is None:
            return

        current_pos = float(center_x if self.orientation == 'vertical' else center_y)

        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = {'last_pos': current_pos, 'counted': False}
            return

        last_pos = float(self.tracked_objects[track_id]['last_pos'])

        if not self.tracked_objects[track_id]['counted']:
            is_forward = last_pos < current_pos
            crossed = False
            count_this = None

            # Detecta cruce por cambio de signo
            if (last_pos - self.line_pos) * (current_pos - self.line_pos) < 0:
                if self.direction is None or \
                   (self.direction == 'left_to_right' and is_forward) or \
                   (self.direction == 'right_to_left' and not is_forward):
                    crossed = True
                    count_this = 'forward' if is_forward else 'backward'

            # Fallback con margen
            elif abs(current_pos - self.line_pos) <= self.margin and abs(current_pos - last_pos) >= self.min_movement:
                if self.direction is None or \
                   (self.direction == 'left_to_right' and last_pos < self.line_pos <= current_pos) or \
                   (self.direction == 'right_to_left' and last_pos > self.line_pos >= current_pos):
                    crossed = True
                    count_this = 'forward' if is_forward else 'backward'

            if crossed and count_this:
                if count_this == 'forward':
                    self.count_forward += 1
                else:
                    self.count_backward += 1
                self.tracked_objects[track_id]['counted'] = True

        self.tracked_objects[track_id]['last_pos'] = current_pos

    def draw(self, frame, color=(0, 255, 0), label_y_start=30, line_start=0.0, line_end=1.0):
        """
        Args:
            line_start: Posición relativa de inicio (0.0 a 1.0)
            line_end: Posición relativa de fin (0.0 a 1.0)
        """
        if self.line_pos is None:
            return

        height, width = frame.shape[:2]

        if self.orientation == 'horizontal':
            y = int(self.line_pos)
            x_start = int(width * line_start)
            x_end = int(width * line_end)
            cv2.line(frame, (x_start, y), (x_end, y), color, 3)
            cv2.putText(frame, f'Arriba->Abajo: {self.count_forward}', (10, label_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f'Abajo->Arriba: {self.count_backward}', (10, label_y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:  # vertical
            x = int(self.line_pos)
            y_start = int(height * line_start)
            y_end = int(height * line_end)
            cv2.line(frame, (x, y_start), (x, y_end), color, 3)
            if self.direction == 'left_to_right':
                cv2.putText(frame, f'Izq->Der: {self.count_forward}', (10, label_y_start),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            elif self.direction == 'right_to_left':
                cv2.putText(frame, f'Der->Izq: {self.count_backward}', (10, label_y_start),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)