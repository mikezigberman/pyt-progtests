# Homeworks

During the semester, a total of 4 programming homework assignments will be assigned. The submission deadline is two weeks from the entry.

Fixing/reviewing tasks is done completely automatically using Gitlab CI, which runs two sets of tests. The first set of tests is available in the repository and to students for local verification of the functionality of the solution. The second (non-public; inaccessible to students) set of tests is used to award points for solving homework.

Submitted homework solutions are regularly checked for plagiarism. Any attempt to submit a third-party, collective or machine-generated homework solution will be considered an attempt to cheat. Such a task will naturally be evaluated in the range of -40 to 0 points for all actors, and this is fully within the competence of the trainee. The whole situation will be further resolved according to the applicable regulations of FIT and CTU.

## Assigned tasks

| Task | will be made available | submission to |
|---|---|---|
| [**Tree visualization**](https://github.com/mikezigberman/pyt-progtests/tree/main/homework01) | 20/03/2023 | 2/04/2023 11:59:59 PM |
| [**Image filtering**](https://github.com/mikezigberman/pyt-progtests/tree/main/homework02) | 	3/04/2023 | 16/04/2023 11:59:59 PM |
| [**ORB detector**](https://github.com/mikezigberman/pyt-progtests/tree/main/homework03) | 17/04/2023 | 30/04/2023 11:59:59 PM |
| [**Data processing**](https://github.com/mikezigberman/pyt-progtests/tree/main/homework04) | 1/05/2023| 14/05/2023 11:59:59 PM |

## Instructions for developing assignments

* Always work out the task independently.
* Your code must be PEP8 compliant.
  * This is checked automatically by pylint . For self-verification, use either your IDE or e.g.: `pylint ./yourcode.py --disable=C0301,C0103`
  * Exceptions are long lines (C0301) and short variable names (C0103).
* Assignments will be submitted through the faculty's Gitlab , where you will also find detailed instructions.
  * No need to make a merge request.
  * If you want a code-review, make an agreement with your trainer individually.

## Referrals and unsolicited good advice

* Test yourself, on an ungraded assignment, that you know how to submit the assignment.

* Read the assignment as soon as possible after it is made available to make sure you understand everything. If you have any questions, ask during the exercises or ask your colleagues - e.g. in the appropriate channel on teams/discord.

* Think about how to solve the problem and start with the parts where you know how to do it.

* Use locally `pytest` instead of running CI on Gitlab.

* Work on the task continuously.

  * Version your work in Git. It can help you prove that you worked on the assignment independently.

  * If you leave it right before the deadline, you increase the risk that the CI will be busy and you will have to wait longer for the evaluation. In the extreme case, it may happen that your assignment will not be evaluated in time and you will not receive points for it.
