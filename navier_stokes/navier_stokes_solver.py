"""
Численный решатель для 2D несжимаемых уравнений Навье-Стокса на
периодической сетке -- спектральный метод (Фурье). Это "компиляция"
дерева из navier_stokes_tree.py в исполняемую numpy-функцию: ровно та же
роль, которую для Volterra-Lotka играл sympy.lambdify, только написанная
руками, т.к. (а) sympy в этой песочнице недоступен, (б) даже при его
наличии скалярная компиляция не годится для полей на сетке -- см.
докстринг navier_stokes_tree.py.

Метод: pseudo-spectral. Производные по x,y считаются точно через
преобразование Фурье (домножение на i*kx, i*ky в частотной области),
нелинейный адвективный член считается в физическом пространстве
(произведение полей), проекция давления убирает дивергентную часть поля
скорости за один шаг в частотной области (решение уравнения Пуассона
для Фурье-образов -- тривиально, т.к. там это просто деление на -k^2).

Интегрирование по времени -- RK4, как и во всём остальном эксперименте.
"""

import numpy as np


class NavierStokes2D:
    def __init__(self, n=64, L=2 * np.pi):
        self.n = n
        self.L = L
        self.dx = L / n

        k1d = 2 * np.pi * np.fft.fftfreq(n, d=self.dx)
        self.kx, self.ky = np.meshgrid(k1d, k1d, indexing="ij")
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0] = 1.0  # чтобы не делить на ноль на нулевой гармонике

    def ddx(self, f):
        return np.real(np.fft.ifft2(1j * self.kx * np.fft.fft2(f)))

    def ddy(self, f):
        return np.real(np.fft.ifft2(1j * self.ky * np.fft.fft2(f)))

    def laplacian(self, f):
        return np.real(np.fft.ifft2(-self.k2 * np.fft.fft2(f)))

    def project_divergence_free(self, vx, vy):
        """
        Проекция Чорина в Фурье-пространстве: убирает градиентную
        (потенциальную) часть поля скорости, оставляя div-free часть.
        Это и есть единственная НЕЛОКАЛЬНАЯ операция дерева
        (pressure_grad_x/y) -- реализована именно здесь, целиком.
        """
        vx_hat = np.fft.fft2(vx)
        vy_hat = np.fft.fft2(vy)
        div_hat = 1j * self.kx * vx_hat + 1j * self.ky * vy_hat
        # давление (в Фурье-пространстве) находится из div(v)=0 condition
        p_hat = div_hat / (-self.k2_safe)
        p_hat[0, 0] = 0.0
        dpdx_hat = 1j * self.kx * p_hat
        dpdy_hat = 1j * self.ky * p_hat
        vx_hat_new = vx_hat - dpdx_hat
        vy_hat_new = vy_hat - dpdy_hat
        return np.real(np.fft.ifft2(vx_hat_new)), np.real(np.fft.ifft2(vy_hat_new))

    def rhs(self, vx, vy, nu):
        """
        Правая часть уравнения момента (без члена давления -- он учитывается
        отдельно проекцией после каждого шага RK4, что эквивалентно
        схеме с расщеплением по физическим процессам, стандартной для
        incompressible NS).
        """
        dvx_dx, dvx_dy = self.ddx(vx), self.ddy(vx)
        dvy_dx, dvy_dy = self.ddx(vy), self.ddy(vy)

        adv_x = vx * dvx_dx + vy * dvx_dy
        adv_y = vx * dvy_dx + vy * dvy_dy

        visc_x = nu * self.laplacian(vx)
        visc_y = nu * self.laplacian(vy)

        return -adv_x + visc_x, -adv_y + visc_y

    def step_rk4(self, vx, vy, nu, dt):
        k1x, k1y = self.rhs(vx, vy, nu)
        k2x, k2y = self.rhs(vx + 0.5 * dt * k1x, vy + 0.5 * dt * k1y, nu)
        k3x, k3y = self.rhs(vx + 0.5 * dt * k2x, vy + 0.5 * dt * k2y, nu)
        k4x, k4y = self.rhs(vx + dt * k3x, vy + dt * k3y, nu)

        vx_new = vx + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vy_new = vy + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        # проекция после каждого шага -- держим поле несжимаемым
        vx_new, vy_new = self.project_divergence_free(vx_new, vy_new)
        return vx_new, vy_new

    def simulate(self, vx0, vy0, nu, dt, n_steps, save_every=1):
        """Возвращает траекторию (n_saved, n, n) для vx и vy."""
        vx, vy = self.project_divergence_free(vx0.copy(), vy0.copy())
        traj_x, traj_y = [vx.copy()], [vy.copy()]

        for step in range(1, n_steps + 1):
            vx, vy = self.step_rk4(vx, vy, nu, dt)
            if step % save_every == 0:
                traj_x.append(vx.copy())
                traj_y.append(vy.copy())

        return np.array(traj_x), np.array(traj_y)

    def kinetic_energy(self, vx, vy):
        """Полная кинетическая энергия поля -- удобный скалярный диагностический критерий для LM."""
        return 0.5 * np.mean(vx ** 2 + vy ** 2)


def make_shear_layer_ic(n=64, L=2 * np.pi, delta=0.05, amplitude=0.01, seed=0):
    """
    Стандартное начальное условие для задачи shear-layer / Kelvin-Helmholtz
    (та же конфигурация, что в датасете The Well "shear_flow"): две
    противоположно направленные полосы течения с тонким слоем сдвига,
    плюс небольшое случайное возмущение, запускающее неустойчивость.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, n, endpoint=False)
    y = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    vx0 = np.tanh((Y - L / 2) / delta) - np.tanh((Y - 3 * L / 2) / delta) - 1.0
    vy0 = amplitude * np.sin(2 * np.pi * X / L) + amplitude * rng.standard_normal((n, n)) * 0.1

    return vx0, vy0


if __name__ == "__main__":
    solver = NavierStokes2D(n=64)
    vx0, vy0 = make_shear_layer_ic(n=64)
    traj_x, traj_y = solver.simulate(vx0, vy0, nu=0.01, dt=0.01, n_steps=200, save_every=20)
    print("Траектория посчитана, форма:", traj_x.shape)
    for i in range(traj_x.shape[0]):
        E = solver.kinetic_energy(traj_x[i], traj_y[i])
        print(f"  шаг {i*20:4d}: кинетическая энергия = {E:.6f}")
