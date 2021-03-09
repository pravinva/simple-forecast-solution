// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./Table.module.scss";
import MaterialTable, { TableProps } from "@material-ui/core/Table";
import MaterialTableBody from "@material-ui/core/TableBody";
import MaterialTableCell, { TableCellProps } from "@material-ui/core/TableCell";
import MaterialTableContainer from "@material-ui/core/TableContainer";
import MaterialTableHead from "@material-ui/core/TableHead";
import MaterialTablePagination from "@material-ui/core/TablePagination";
import MaterialTableRow from "@material-ui/core/TableRow";
import MaterialTableSortLabel from "@material-ui/core/TableSortLabel";
import classNames from "classnames";

const Container: React.FC<TableProps> = ({ children, className, ...props }) => {
  return (
    <div className={classNames(className, styles.container)}>
      <MaterialTableContainer>
        <MaterialTable {...props}>{children}</MaterialTable>
      </MaterialTableContainer>
    </div>
  );
};

const Cell: React.FC<TableCellProps> = ({ children, ...props }) => {
  return (
    <MaterialTableCell className={styles.cell} {...props}>
      <div className={styles.cellInner}>{children}</div>
    </MaterialTableCell>
  );
};

const Table = {
  Container,
  Body: MaterialTableBody,
  Cell,
  Head: MaterialTableHead,
  Pagination: MaterialTablePagination,
  Row: MaterialTableRow,
  SortLabel: MaterialTableSortLabel,
};

export default Table;
