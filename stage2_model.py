"""
ЕТАП 2. Імітаційна модель БАГАТОКАНАЛЬНОЇ системи (n стоматологів, спільна черга).
Розширення моделі етапу 1: замість одного лікаря – список з n лікарів.
"""
import numpy as np
import pandas as pd

from params import (N_STEPS, HOURS_PER_DAY, WORK_HOURS, QUEUE_LIMIT, SALARY, DAYS, N_RUNS,
                    SERVICES, lam_at, outage_plan, new_client)


def simulate(n_channels, rng, queue_limit=QUEUE_LIMIT, const_lambda=None, outages=True,
             log=False):
    """Один прогін моделі з n_channels стоматологами (при n=1 збігається з етапом 1)."""
    plan = outage_plan(rng) if outages else np.zeros(N_STEPS, dtype=bool)
    queue = []
    doctors = [None] * n_channels     # для кожного лікаря: [залишок часу, послуга] або None
    served = lost = revenue = lost_revenue = 0
    idle_time = 0.0
    waits, rows = [], []

    for step in range(N_STEPS):
        hour = WORK_HOURS[step % HOURS_PER_DAY]
        lam = const_lambda if const_lambda is not None else lam_at(hour)
        off = bool(plan[step])

        # ---- надходження клієнтів
        k = rng.poisson(lam)
        lost_step = 0
        for _ in range(k):
            t_arr, s, dur = new_client(rng, step)
            all_busy = all(d is not None for d in doctors)
            if len(queue) >= queue_limit and (all_busy or off):
                lost += 1
                lost_step += 1
                lost_revenue += SERVICES["Ціна"].iat[s]
            else:
                queue.append((t_arr, s, dur))
        queue.sort(key=lambda x: x[0])

        # ---- обслуговування: кожен лікар працює 1 год
        served_step = 0
        if not off:
            for i in range(n_channels):
                t, end = float(step), step + 1.0
                while t < end:
                    if doctors[i] is None:
                        if queue and queue[0][0] <= end:
                            t_arr, s, dur = queue.pop(0)
                            if t_arr > t:
                                idle_time += t_arr - t
                                t = t_arr
                            waits.append(t - t_arr)
                            doctors[i] = [dur, s]
                        else:
                            idle_time += end - t
                            break
                    work = min(doctors[i][0], end - t)
                    t += work
                    doctors[i][0] -= work
                    if doctors[i][0] <= 1e-9:
                        served += 1
                        served_step += 1
                        revenue += SERVICES["Ціна"].iat[doctors[i][1]]
                        doctors[i] = None

        if log:
            rows.append({"Крок": step + 1, "День": step // HOURS_PER_DAY + 1,
                         "Година": f"{hour:02d}:00", "lambda": lam,
                         "Відключення": "так" if off else "", "Надійшло": k,
                         "Втрачено": lost_step, "Обслуговано": served_step,
                         "Черга": len(queue),
                         "Зайнято лікарів": sum(d is not None for d in doctors)})

    work_time = n_channels * (N_STEPS - plan.sum())
    total = served + lost + len(queue) + sum(d is not None for d in doctors)
    cost = n_channels * SALARY * DAYS
    res = {"Стоматологів": n_channels, "Клієнтів": total, "Обслуговано": served,
           "Втрачено": lost, "P_відмови": lost / total if total else 0,
           "Сер. очікування, год": np.mean(waits) if waits else 0,
           "P_простою": idle_time / work_time if work_time else 0,
           "Завантаженість": 1 - idle_time / work_time if work_time else 0,
           "Годин відключення": int(plan.sum()),
           "Дохід": revenue, "Зарплата": cost, "Недоотримано": lost_revenue,
           "Прибуток": revenue - cost}
    return (res, pd.DataFrame(rows)) if log else res


def monte_carlo(n_channels, runs=N_RUNS, seed=0, **kw):
    """Метод статистичних випробувань: runs прогонів моделі."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([simulate(n_channels, rng, **kw) for _ in range(runs)])


if __name__ == "__main__":
    # перевірка: при n=1 результат збігається з моделлю етапу 1
    from stage1_model import simulate_one
    print(simulate(1, np.random.default_rng(5)))
    print(simulate_one(np.random.default_rng(5)))
